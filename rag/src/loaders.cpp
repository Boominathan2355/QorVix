#include "qorvix/rag/loaders.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace qorvix::rag {

namespace {

std::string lowerExt(const fs::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

bool readFile(const fs::path& path, std::string& out, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  // Strip a UTF-8 BOM: left in place it becomes a leading [UNK] on every first chunk.
  if (out.size() >= 3 && static_cast<unsigned char>(out[0]) == 0xEF &&
      static_cast<unsigned char>(out[1]) == 0xBB && static_cast<unsigned char>(out[2]) == 0xBF) {
    out.erase(0, 3);
  }
  // Normalize CRLF so byte offsets and boundary detection behave the same on both platforms.
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  return true;
}

// Strips Markdown syntax while KEEPING the text it decorates. Retrieval quality depends on the
// prose, and leaving '#', '*' and link brackets in place both wastes tokens and distorts the
// embedding — but dropping link TEXT would lose real content, so only the URL part goes.
std::string stripMarkdown(const std::string& src) {
  std::string out;
  out.reserve(src.size());
  bool inFence = false;

  std::istringstream lines(src);
  std::string line;
  while (std::getline(lines, line)) {
    const std::string trimmed = line.substr(line.find_first_not_of(" \t") == std::string::npos
                                                ? line.size()
                                                : line.find_first_not_of(" \t"));
    if (trimmed.rfind("```", 0) == 0 || trimmed.rfind("~~~", 0) == 0) {
      inFence = !inFence;
      out.push_back('\n');
      continue;
    }
    if (inFence) {
      // Keep fenced code as-is: in technical docs the code IS the answer to many queries.
      out += line;
      out.push_back('\n');
      continue;
    }

    std::string work = line;
    // ATX headings: drop the leading '#'s, keep the heading text.
    std::size_t h = 0;
    while (h < work.size() && work[h] == '#') ++h;
    if (h > 0 && h <= 6 && h < work.size() && work[h] == ' ') work = work.substr(h + 1);
    // Setext underlines carry no text of their own.
    if (!work.empty() && work.find_first_not_of("=-") == std::string::npos && work.size() > 2) {
      continue;
    }
    // Blockquote and list markers.
    std::size_t i = 0;
    while (i < work.size() && (work[i] == '>' || work[i] == ' ' || work[i] == '\t')) ++i;
    work = work.substr(i);
    if (work.size() > 1 && (work[0] == '-' || work[0] == '*' || work[0] == '+') && work[1] == ' ') {
      work = work.substr(2);
    }

    // Inline: [text](url) -> text, and drop emphasis/backtick runs.
    std::string cleaned;
    cleaned.reserve(work.size());
    for (std::size_t j = 0; j < work.size(); ++j) {
      if (work[j] == '[') {
        const std::size_t close = work.find(']', j);
        if (close != std::string::npos && close + 1 < work.size() && work[close + 1] == '(') {
          const std::size_t end = work.find(')', close);
          if (end != std::string::npos) {
            cleaned += work.substr(j + 1, close - j - 1);
            j = end;
            continue;
          }
        }
      }
      if (work[j] == '*' || work[j] == '_' || work[j] == '`') continue;
      cleaned.push_back(work[j]);
    }
    out += cleaned;
    out.push_back('\n');
  }
  return out;
}

// RFC-4180-ish: quoted fields, "" escapes, embedded separators and newlines. One row per record,
// fields joined by " | " so a row reads as a sentence to the encoder rather than as raw CSV.
std::string flattenDelimited(const std::string& src, char sep) {
  std::string out;
  std::string field;
  std::vector<std::string> row;
  bool inQuotes = false;

  auto endRow = [&] {
    if (!field.empty() || !row.empty()) row.push_back(field);
    field.clear();
    bool any = false;
    for (const auto& f : row) {
      if (f.empty()) continue;
      if (any) out += " | ";
      out += f;
      any = true;
    }
    if (any) out.push_back('\n');
    row.clear();
  };

  for (std::size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    if (inQuotes) {
      if (c == '"') {
        if (i + 1 < src.size() && src[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          inQuotes = false;
        }
      } else {
        field.push_back(c);
      }
    } else if (c == '"') {
      inQuotes = true;
    } else if (c == sep) {
      row.push_back(field);
      field.clear();
    } else if (c == '\n') {
      endRow();
    } else {
      field.push_back(c);
    }
  }
  endRow();
  return out;
}

}  // namespace

bool isSupportedExtension(const std::string& ext) {
  return ext == ".txt" || ext == ".md" || ext == ".markdown" || ext == ".text" || ext == ".csv" ||
         ext == ".tsv";
}

std::optional<Document> loadDocument(const fs::path& path, std::string& error) {
  error.clear();
  const std::string ext = lowerExt(path);
  if (ext == ".pdf" || ext == ".docx" || ext == ".doc") {
    error = ext.substr(1) + " loading is not implemented yet — it needs a DEFLATE decoder first "
                            "(see rag/loaders.hpp)";
    return std::nullopt;
  }
  if (!isSupportedExtension(ext)) {
    error = "no loader for '" + ext + "'";
    return std::nullopt;
  }

  std::string raw;
  if (!readFile(path, raw, error)) return std::nullopt;

  Document doc;
  doc.id = path.generic_string();
  doc.source = path.generic_string();
  if (ext == ".md" || ext == ".markdown") {
    doc.text = stripMarkdown(raw);
  } else if (ext == ".csv") {
    doc.text = flattenDelimited(raw, ',');
  } else if (ext == ".tsv") {
    doc.text = flattenDelimited(raw, '\t');
  } else {
    doc.text = std::move(raw);
  }
  return doc;
}

std::vector<Document> loadDirectory(const fs::path& dir, std::vector<std::string>& skipped) {
  std::vector<Document> docs;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    // A single file is a legitimate target too; treating it as a one-document directory keeps
    // the CLI from needing two code paths.
    std::string err;
    if (auto d = loadDocument(dir, err)) docs.push_back(std::move(*d));
    else skipped.push_back(dir.generic_string() + ": " + err);
    return docs;
  }

  std::vector<fs::path> paths;
  for (fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec),
       end;
       it != end; it.increment(ec)) {
    if (ec) break;
    if (it->is_regular_file(ec)) paths.push_back(it->path());
  }
  // Deterministic order so an index built twice from the same tree is identical.
  std::sort(paths.begin(), paths.end());

  for (const auto& p : paths) {
    if (!isSupportedExtension(lowerExt(p))) continue;
    std::string err;
    if (auto d = loadDocument(p, err)) docs.push_back(std::move(*d));
    else skipped.push_back(p.generic_string() + ": " + err);
  }
  return docs;
}

}  // namespace qorvix::rag
