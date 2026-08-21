---
name: code-review
version: 1.0.0
description: Comprehensive code review, vulnerability audit, and performance optimization
category: coding
author: QorVix
tags: ["code", "review", "security", "cpp", "performance"]
required_tools: ["file_ops"]
---

# Code Review Playbook

When conducting an automated or assisted code review, follow this systematic procedure:

## 1. Safety & Memory Correctness
- Check for buffer overflows, out-of-bounds array access, and unchecked pointer arithmetic.
- Verify RAII ownership: no raw `delete` or dangling pointers.
- Look for concurrency data races, unprotected shared state, and missing mutex locks.
- Ensure all error return codes or exception paths cleanly unwind resources.

## 2. Modern C++23 Best Practices
- Enforce `std::string_view` for read-only strings and `std::span` for array slices.
- Verify `const`, `constexpr`, and `noexcept` specifications on move constructors.
- Check structured bindings and range-based loops.

## 3. Performance & Efficiency
- Avoid extraneous heap allocations inside tight inference/decoding loops.
- Check cache locality and struct memory alignment.
- Identify SIMD (AVX2/AVX-512) and GPU acceleration candidates.

## 4. Structured Output Format
Categorize review findings clearly:
- **[CRITICAL]**: Security vulnerabilities, memory leaks, undefined behavior.
- **[MAJOR]**: Correctness flaws, logic regressions, race conditions.
- **[MINOR]**: Performance inefficiencies, redundant copies.
- **[STYLE]**: Naming conventions, documentation, code clarity.
