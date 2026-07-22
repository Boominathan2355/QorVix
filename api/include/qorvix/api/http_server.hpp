#pragma once

#include <cstdint>
#include <functional>
#include <string>

// A minimal, dependency-free HTTP/1.1 server (cross-platform BSD/Winsock sockets). Blocking,
// one connection at a time, `Connection: close` per response — enough to expose the OpenAI API
// without pulling in Boost/Asio, keeping the "build from scratch" mission and the vcpkg-free
// build. (Boost.Beast remains the option for production hardening / high concurrency.)
namespace qorvix::api {

struct HttpRequest {
  std::string method;   // "GET", "POST", ...
  std::string target;   // path, e.g. "/v1/chat/completions"
  std::string body;     // request body (read via Content-Length)
};

// Passed to the handler to produce the response. Either send() one complete response, or
// beginStream()/writeChunk()/endStream() for SSE. Exactly one of the two modes per request.
class HttpResponder {
 public:
  void send(int status, const std::string& contentType, const std::string& body);

  void beginStream(int status, const std::string& contentType);  // e.g. "text/event-stream"
  void writeChunk(const std::string& data);                      // raw bytes (caller frames SSE)
  void endStream();

  bool responded() const noexcept { return responded_; }

 private:
  friend class HttpServer;
  explicit HttpResponder(std::uint64_t sock) : sock_(sock) {}
  std::uint64_t sock_;
  bool responded_ = false;
  bool streaming_ = false;
};

using HttpHandler = std::function<void(const HttpRequest&, HttpResponder&)>;

class HttpServer {
 public:
  explicit HttpServer(int port);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds and listens. Returns false with `error` set on failure.
  bool start(std::string& error);
  // Accept loop — blocks until stop() is called (from another thread / signal handler).
  void run(const HttpHandler& handler);
  void stop();

  int port() const noexcept { return port_; }

 private:
  int port_;
  std::uint64_t listen_ = ~0ull;  // invalid socket sentinel
  bool running_ = false;
};

}  // namespace qorvix::api
