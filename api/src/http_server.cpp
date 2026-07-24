#include "qorvix/api/http_server.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t kInvalid = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t kInvalid = -1;
#endif

#include <thread>

namespace qorvix::api {

namespace {

void closeSock(socket_t s) {
#if defined(_WIN32)
  ::closesocket(s);
#else
  ::close(s);
#endif
}

bool sendAll(socket_t s, const char* data, std::size_t len) {
  std::size_t sent = 0;
  while (sent < len) {
    const int n = ::send(s, data + sent, static_cast<int>(len - sent), 0);
    if (n <= 0) return false;
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

bool sendAll(socket_t s, const std::string& str) { return sendAll(s, str.data(), str.size()); }

const char* statusText(int code) {
  switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    default: return "OK";
  }
}

// Reads one HTTP request (request line + headers + Content-Length body). Returns false on a
// closed/broken connection or a malformed head.
bool readRequest(socket_t s, HttpRequest& req) {
  std::string buf;
  char chunk[4096];
  std::size_t headerEnd = std::string::npos;

  // Read until the end of headers.
  while (headerEnd == std::string::npos) {
    const int n = ::recv(s, chunk, sizeof(chunk), 0);
    if (n <= 0) return false;
    buf.append(chunk, static_cast<std::size_t>(n));
    headerEnd = buf.find("\r\n\r\n");
    if (buf.size() > (1u << 20)) return false;  // 1 MB header cap
  }

  const std::string head = buf.substr(0, headerEnd);
  const std::size_t lineEnd = head.find("\r\n");
  const std::string requestLine = head.substr(0, lineEnd);

  // "METHOD SP TARGET SP VERSION"
  const std::size_t sp1 = requestLine.find(' ');
  const std::size_t sp2 = requestLine.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  req.method = requestLine.substr(0, sp1);
  req.target = requestLine.substr(sp1 + 1, sp2 - sp1 - 1);

  // Content-Length (case-insensitive header scan).
  std::size_t contentLength = 0;
  std::string lower = head;
  for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (const std::size_t p = lower.find("content-length:"); p != std::string::npos) {
    contentLength = static_cast<std::size_t>(std::strtoul(lower.c_str() + p + 15, nullptr, 10));
  }

  // Body already partially in buf after the header terminator.
  std::string body = buf.substr(headerEnd + 4);
  while (body.size() < contentLength) {
    const int n = ::recv(s, chunk, sizeof(chunk), 0);
    if (n <= 0) break;
    body.append(chunk, static_cast<std::size_t>(n));
  }
  req.body = std::move(body);
  return true;
}

}  // namespace

// ---- HttpResponder -------------------------------------------------------------------------

void HttpResponder::send(int status, const std::string& contentType, const std::string& body) {
  if (responded_) return;
  responded_ = true;
  std::string head = "HTTP/1.1 " + std::to_string(status) + " " + statusText(status) + "\r\n";
  head += "Content-Type: " + contentType + "\r\n";
  head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  head += "Connection: close\r\n\r\n";
  sendAll(static_cast<socket_t>(sock_), head);
  sendAll(static_cast<socket_t>(sock_), body);
}

void HttpResponder::beginStream(int status, const std::string& contentType) {
  if (responded_) return;
  responded_ = true;
  streaming_ = true;
  std::string head = "HTTP/1.1 " + std::to_string(status) + " " + statusText(status) + "\r\n";
  head += "Content-Type: " + contentType + "\r\n";
  head += "Cache-Control: no-cache\r\n";
  head += "Access-Control-Allow-Origin: *\r\n";
  head += "Connection: close\r\n\r\n";  // client reads until EOF
  sendAll(static_cast<socket_t>(sock_), head);
}

void HttpResponder::writeChunk(const std::string& data) {
  if (streaming_) sendAll(static_cast<socket_t>(sock_), data);
}

void HttpResponder::endStream() { /* connection is closed by the accept loop */ }

// ---- HttpServer ----------------------------------------------------------------------------

HttpServer::HttpServer(int port) : port_(port) {
#if defined(_WIN32)
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

HttpServer::~HttpServer() {
  stop();
#if defined(_WIN32)
  WSACleanup();
#endif
}

bool HttpServer::start(std::string& error) {
  socket_t s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s == kInvalid) {
    error = "socket() failed";
    return false;
  }
  int yes = 1;
  ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<std::uint16_t>(port_));

  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    error = "bind() failed on port " + std::to_string(port_);
    closeSock(s);
    return false;
  }
  if (::listen(s, 16) != 0) {
    error = "listen() failed";
    closeSock(s);
    return false;
  }
  listen_ = static_cast<std::uint64_t>(s);
  return true;
}

void HttpServer::run(const HttpHandler& handler) {
  running_ = true;
  const socket_t ls = static_cast<socket_t>(listen_);
  while (running_) {
    socket_t client = ::accept(ls, nullptr, nullptr);
    if (client == kInvalid) {
      if (!running_) break;
      continue;
    }
    std::thread([client, handler] {
      HttpRequest req;
      if (readRequest(client, req)) {
        HttpResponder responder(static_cast<std::uint64_t>(client));
        handler(req, responder);
        if (!responder.responded()) {
          responder.send(500, "text/plain", "no response produced\n");
        }
      }
      closeSock(client);
    }).detach();
  }
}

void HttpServer::stop() {
  running_ = false;
  if (listen_ != ~0ull) {
    closeSock(static_cast<socket_t>(listen_));
    listen_ = ~0ull;
  }
}

}  // namespace qorvix::api
