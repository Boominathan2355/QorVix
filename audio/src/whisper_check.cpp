#include "qorvix/audio/whisper_check.hpp"

#include <fstream>
#include <sstream>

namespace qorvix::audio {

namespace {

// Fixtures are committed as text and checked out with the platform's line endings, so a stray
// carriage return would otherwise end up inside the transcript this file compares byte for byte.
void stripCr(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
}

}  // namespace

bool WhisperReference::load(const std::filesystem::path& path, std::string& error) {
  error.clear();
  std::ifstream in(path);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    stripCr(line);
    if (line.empty() || line[0] == '#') continue;

    // `text` is taken as the raw remainder of the line: the transcript's own leading space is part
    // of what Whisper produced, and a stream-extracted first token would eat it.
    if (line.rfind("text ", 0) == 0) {
      text = line.substr(5);
      std::string decoded;
      decoded.reserve(text.size());
      for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 1 < text.size() && text[i + 1] == 'n') {
          decoded.push_back('\n');
          ++i;
        } else {
          decoded.push_back(text[i]);
        }
      }
      text = decoded;
      continue;
    }

    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "model") {
      ls >> model;
    } else if (key == "language") {
      ls >> language;
    } else if (key == "d_model") {
      ls >> dModel;
    } else if (key == "enc_ctx") {
      ls >> encCtx;
    } else if (key == "vocab") {
      ls >> vocab;
    } else if (key == "max_new_tokens") {
      ls >> maxNewTokens;
    } else if (key == "argmax0") {
      ls >> argmax0;
    } else if (key == "prompt") {
      int id = 0;
      while (ls >> id) prompt.push_back(id);
    } else if (key == "tokens") {
      int id = 0;
      while (ls >> id) tokens.push_back(id);
    } else if (key == "enc_frame0") {
      float v = 0.0f;
      while (ls >> v) encFrame0.push_back(v);
    } else if (key == "enc_dim_means") {
      float v = 0.0f;
      while (ls >> v) encDimMeans.push_back(v);
    } else if (key == "logits_top") {
      LogitProbe p;
      while (ls >> p.id >> p.value) logitsTop.push_back(p);
    } else if (key == "logit_probe") {
      LogitProbe p;
      ls >> p.id >> p.value;
      logitProbes.push_back(p);
    }
  }

  if (dModel <= 0 || encCtx <= 0 || vocab <= 0 || prompt.empty() || encFrame0.empty() ||
      logitsTop.empty() || tokens.empty()) {
    error = "malformed whisper fixture (d_model " + std::to_string(dModel) + " enc_ctx " +
            std::to_string(encCtx) + " vocab " + std::to_string(vocab) + " prompt " +
            std::to_string(prompt.size()) + " enc_frame0 " + std::to_string(encFrame0.size()) +
            " logits_top " + std::to_string(logitsTop.size()) + " tokens " +
            std::to_string(tokens.size()) + ")";
    return false;
  }
  // The header and the payload have to agree, or a fixture captured on one model could be compared
  // against another's forward pass and fail as a numerical mismatch rather than the wrong-pairing
  // mistake it is.
  if (static_cast<int>(encFrame0.size()) != dModel ||
      (!encDimMeans.empty() && static_cast<int>(encDimMeans.size()) != dModel)) {
    error = "whisper fixture encoder rows disagree with its own d_model";
    return false;
  }
  return true;
}

}  // namespace qorvix::audio
