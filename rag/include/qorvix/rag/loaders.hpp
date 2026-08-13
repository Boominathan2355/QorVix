#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/rag/document.hpp"

namespace qorvix::rag {

// Loads one document, dispatching on the file extension. Returns nullopt with `error` set.
std::optional<Document> loadDocument(const std::filesystem::path& path, std::string& error);

// Loads every supported file under `dir` (recursively), skipping unsupported extensions rather
// than failing the whole run. Files that ARE supported but fail to read are reported in `skipped`.
std::vector<Document> loadDirectory(const std::filesystem::path& dir,
                                    std::vector<std::string>& skipped);

// True if this extension has a loader. Used by loadDirectory and by the CLI's summary.
bool isSupportedExtension(const std::string& ext);

// ---- format notes ---------------------------------------------------------------------------
//
// Supported: .txt, .md/.markdown, .csv/.tsv.
//
// PDF and DOCX are NOT supported, and the reason is worth stating rather than leaving as a gap.
// A correct PDF text extractor needs an object/xref parser, FlateDecode (i.e. zlib — a dependency
// this module does not have, or a from-scratch inflate), CMap/ToUnicode font mapping, and
// text-position reassembly into reading order. DOCX needs a ZIP reader (inflate again) plus XML.
// Each is a multi-week subproject with nothing to do with inference.
//
// Doing them BADLY is worse than not doing them: naive BT/ET scraping yields garbage text that
// silently poisons every embedding derived from it. A missing loader errors; a bad loader lies.
// The honest prerequisite is a from-scratch DEFLATE decoder (~400 lines, exactly specified by
// RFC 1951 and testable against known vectors), which would unlock both — its own work item.

}  // namespace qorvix::rag
