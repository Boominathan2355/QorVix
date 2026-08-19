#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <vector>

#include "qorvix/api/multipart.hpp"

using namespace qorvix::api;

// Phase 11b-3b: multipart/form-data, the one OpenAI route that is not JSON. This parser is
// reachable by anyone who can reach the port, so the tests are mostly about what it REFUSES —
// a form parser that accepts a truncated upload transcribes noise and reports success.

namespace {

const std::string kBoundary = "----qorvixBoundary7MA4YWxkTrZu0gW";

std::string part(const std::string& disposition, const std::string& extraHeaders,
                 const std::string& content) {
  return "--" + kBoundary + "\r\nContent-Disposition: form-data; " + disposition + "\r\n" +
         extraHeaders + "\r\n" + content + "\r\n";
}

std::string closing() { return "--" + kBoundary + "--\r\n"; }

std::string contentType() { return "multipart/form-data; boundary=" + kBoundary; }

}  // namespace

TEST_CASE("a well-formed upload parses into its fields", "[multipart]") {
  const std::string body = part("name=\"model\"", "", "whisper-1") +
                           part("name=\"file\"; filename=\"speech.wav\"",
                                "Content-Type: audio/wav\r\n", "RIFFxxxxWAVE") +
                           part("name=\"response_format\"", "", "text") + closing();
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE(parseMultipart(contentType(), body, parts, err));
  REQUIRE(err.empty());
  REQUIRE(parts.size() == 3);

  const FormPart* file = findPart(parts, "file");
  REQUIRE(file != nullptr);
  REQUIRE(file->filename == "speech.wav");
  REQUIRE(file->contentType == "audio/wav");
  REQUIRE(file->content == "RIFFxxxxWAVE");
  REQUIRE(findPart(parts, "model")->content == "whisper-1");
  REQUIRE(findPart(parts, "response_format")->content == "text");
  REQUIRE(findPart(parts, "language") == nullptr);
  // A field is not a file, and saying otherwise would make a route treat one as the other.
  REQUIRE(findPart(parts, "model")->filename.empty());
}

TEST_CASE("part content keeps its bytes exactly", "[multipart]") {
  // The CRLF before a boundary is framing, not content. Keeping it appends two bytes to every
  // uploaded file — which a WAV parser reading a length-prefixed chunk will not notice, and a
  // decoder will read as two samples of noise.
  const std::string audio = std::string("RIFF\0\0\0\0WAVE\r\ndata\r\n", 22);
  const std::string body =
      part("name=\"file\"; filename=\"a.wav\"", "Content-Type: audio/wav\r\n", audio) + closing();
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE(parseMultipart(contentType(), body, parts, err));
  REQUIRE(parts.size() == 1);
  REQUIRE(parts[0].content.size() == audio.size());
  REQUIRE(parts[0].content == audio);
}

TEST_CASE("an empty body between the delimiters is a part, not an error", "[multipart]") {
  const std::string body = part("name=\"file\"; filename=\"empty.wav\"", "", "") + closing();
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE(parseMultipart(contentType(), body, parts, err));
  REQUIRE(parts.size() == 1);
  REQUIRE(parts[0].content.empty());  // the ROUTE decides an empty file is unusable, not the parser
}

TEST_CASE("the boundary parameter may be quoted, and its case is preserved", "[multipart]") {
  const std::string body = part("name=\"file\"", "", "data") + closing();
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE(parseMultipart("multipart/form-data; boundary=\"" + kBoundary + "\"", body, parts, err));
  REQUIRE(parts.size() == 1);
  // Case-insensitive on the parameter NAME only: the boundary itself is case-sensitive, and
  // lowercasing it would fail to find any delimiter at all.
  REQUIRE(parseMultipart("multipart/form-data; BOUNDARY=" + kBoundary, body, parts, err));
  REQUIRE(parts.size() == 1);
  REQUIRE_FALSE(parseMultipart("multipart/form-data; boundary=" + kBoundary + "X", body, parts,
                               err));
}

TEST_CASE("a body that is not multipart is refused with the reason", "[multipart]") {
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE_FALSE(isMultipartFormData("application/json"));
  REQUIRE_FALSE(parseMultipart("application/json", "{}", parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("not multipart/form-data"));

  REQUIRE(isMultipartFormData("multipart/form-data; boundary=x"));
  REQUIRE(isMultipartFormData("Multipart/Form-Data; boundary=x"));
  REQUIRE_FALSE(parseMultipart("multipart/form-data", "whatever", parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("boundary"));
}

TEST_CASE("a truncated upload is refused rather than half-accepted", "[multipart]") {
  std::vector<FormPart> parts;
  std::string err;

  // Cut off mid-content: the closing delimiter never arrives.
  const std::string truncated = "--" + kBoundary +
                                "\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nRIFFxx";
  REQUIRE_FALSE(parseMultipart(contentType(), truncated, parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("unterminated"));
  REQUIRE(parts.empty());

  // Headers that never terminate.
  const std::string noHeaderEnd =
      "--" + kBoundary + "\r\nContent-Disposition: form-data; name=\"file\"\r\n";
  REQUIRE_FALSE(parseMultipart(contentType(), noHeaderEnd, parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("header terminator"));

  // No delimiter anywhere: a JSON body posted with a multipart content type.
  REQUIRE_FALSE(parseMultipart(contentType(), "{\"file\":\"x\"}", parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("no multipart boundary"));
}

TEST_CASE("a body with too many parts is refused", "[multipart]") {
  // The 32 MB request cap does not bound the part COUNT: thousands of tiny fields fit inside it,
  // and each one costs a header parse and a substring copy.
  std::string body;
  for (std::size_t i = 0; i <= kMaxFormParts; ++i) {
    body += part("name=\"f" + std::to_string(i) + "\"", "", "v");
  }
  body += closing();
  std::vector<FormPart> parts;
  std::string err;
  REQUIRE_FALSE(parseMultipart(contentType(), body, parts, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("too many form parts"));
}
