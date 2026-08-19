#include "qorvix/api/multipart.hpp"

#include <algorithm>
#include <cctype>

namespace qorvix::api {

namespace {

std::string toLower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The value of a `key="value"` (or bare `key=value`) parameter inside a header line. Quoted-string
// only — RFC 5987's `filename*=UTF-8''…` encoding is NOT decoded, because a filename this code
// never opens is metadata, and half-decoding it would be worse than reporting it verbatim.
std::string headerParam(const std::string& header, const std::string& key) {
  const std::string lower = toLower(header);
  std::size_t at = lower.find(key + "=");
  if (at == std::string::npos) return {};
  at += key.size() + 1;
  if (at < header.size() && header[at] == '"') {
    const std::size_t end = header.find('"', at + 1);
    if (end == std::string::npos) return {};
    return header.substr(at + 1, end - at - 1);
  }
  const std::size_t end = header.find_first_of(";\r\n", at);
  std::string value = header.substr(at, end == std::string::npos ? std::string::npos : end - at);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
  return value;
}

// One header block ("Name: value" lines) -> the fields a part can carry.
void parsePartHeaders(const std::string& headers, FormPart& part) {
  std::size_t pos = 0;
  while (pos < headers.size()) {
    std::size_t eol = headers.find("\r\n", pos);
    if (eol == std::string::npos) eol = headers.size();
    const std::string line = headers.substr(pos, eol - pos);
    const std::string lower = toLower(line);
    if (lower.rfind("content-disposition:", 0) == 0) {
      part.name = headerParam(line, "name");
      part.filename = headerParam(line, "filename");
    } else if (lower.rfind("content-type:", 0) == 0) {
      std::string value = line.substr(13);
      const std::size_t first = value.find_first_not_of(" \t");
      if (first != std::string::npos) value = value.substr(first);
      part.contentType = value;
    }
    pos = eol + 2;
  }
}

}  // namespace

bool isMultipartFormData(const std::string& contentType) {
  return toLower(contentType).rfind("multipart/form-data", 0) == 0;
}

bool parseMultipart(const std::string& contentType, const std::string& body,
                    std::vector<FormPart>& parts, std::string& error) {
  error.clear();
  parts.clear();
  if (!isMultipartFormData(contentType)) {
    error = "Content-Type is not multipart/form-data";
    return false;
  }
  // The boundary is case-sensitive, so it is read out of the original header text; only the
  // parameter NAME is matched case-insensitively.
  const std::string boundary = headerParam(contentType, "boundary");
  if (boundary.empty()) {
    error = "multipart/form-data without a boundary parameter";
    return false;
  }
  const std::string delim = "--" + boundary;

  std::size_t pos = body.find(delim);
  if (pos == std::string::npos) {
    error = "body contains no multipart boundary";
    return false;
  }
  pos += delim.size();

  while (true) {
    // "--" after a delimiter closes the body; anything else must be CRLF then a part.
    if (body.compare(pos, 2, "--") == 0) return true;
    if (body.compare(pos, 2, "\r\n") != 0) {
      error = "malformed multipart delimiter";
      return false;
    }
    pos += 2;
    const std::size_t headerEnd = body.find("\r\n\r\n", pos);
    if (headerEnd == std::string::npos) {
      error = "multipart part without a header terminator";
      return false;
    }
    FormPart part;
    parsePartHeaders(body.substr(pos, headerEnd - pos + 2), part);
    const std::size_t contentStart = headerEnd + 4;
    // A part ends at the CRLF that precedes the next delimiter — that CRLF belongs to the framing,
    // not to the content, and keeping it would append two bytes to every uploaded file.
    const std::size_t next = body.find("\r\n" + delim, contentStart);
    if (next == std::string::npos) {
      error = "unterminated multipart part";
      return false;
    }
    part.content = body.substr(contentStart, next - contentStart);
    if (parts.size() >= kMaxFormParts) {
      error = "too many form parts (max " + std::to_string(kMaxFormParts) + ")";
      return false;
    }
    parts.push_back(std::move(part));
    pos = next + 2 + delim.size();
    if (pos > body.size()) {
      error = "multipart body ends inside a delimiter";
      return false;
    }
  }
}

const FormPart* findPart(const std::vector<FormPart>& parts, std::string_view name) {
  for (const FormPart& p : parts) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

}  // namespace qorvix::api
