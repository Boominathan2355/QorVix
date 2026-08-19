#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace qorvix::api {

// multipart/form-data parsing (RFC 7578), for the one OpenAI route that is not JSON:
// POST /v1/audio/transcriptions takes the audio as a file upload, which is what every OpenAI SDK
// sends. A JSON-with-base64 variant would have been less code here and incompatible with all of
// them.
//
// Deliberately strict. A form parser is reachable by anyone who can reach the port, so every
// malformed shape is refused with a reason rather than partially accepted: an unterminated part, a
// missing boundary, a missing header terminator. "Parsed most of it" is how an upload ends up
// half-read and transcribed as noise.
struct FormPart {
  std::string name;         // Content-Disposition: form-data; name="..."
  std::string filename;     // ... filename="..."; empty for a plain field
  std::string contentType;  // the part's own Content-Type, if it declared one
  std::string content;      // raw bytes, exactly as they arrived
};

// Caps the number of parts. A transcription request has a handful of fields; a body with hundreds
// is either a mistake or an attempt to make the parser do the work, and the 32 MB body cap does not
// constrain the COUNT by itself.
inline constexpr std::size_t kMaxFormParts = 32;

// `contentType` is the request's Content-Type header, verbatim — the boundary is a parameter of it
// and cannot be recovered from the body alone. Returns false with `error` set.
bool parseMultipart(const std::string& contentType, const std::string& body,
                    std::vector<FormPart>& parts, std::string& error);

// True when this Content-Type is multipart/form-data at all, so a route can answer "send multipart"
// rather than "malformed multipart" to a client that posted JSON.
bool isMultipartFormData(const std::string& contentType);

// The first part with this name, or nullptr. Field names are case-sensitive, as in the spec.
const FormPart* findPart(const std::vector<FormPart>& parts, std::string_view name);

}  // namespace qorvix::api
