---
name: cpp-refactor
version: 1.0.0
description: Modern C++23 refactoring, RAII resource safety, and zero-cost abstractions
category: coding
author: QorVix
tags: ["cpp", "refactor", "modern-cpp", "raii"]
required_tools: ["file_ops"]
---

# Modern C++23 Refactoring Playbook

Guidelines for modernizing and refactoring legacy C/C++ code:

## 1. Eliminate Raw Pointers and Resource Leaks
- Convert manual memory management to `std::unique_ptr` or stack value semantics.
- Encapsulate OS handles (files, sockets, GPU buffers) in RAII wrapper structs.

## 2. Views over Copies
- Use `std::string_view` for string parameters that do not take ownership.
- Use `std::span<const T>` for contiguous buffer inspection.

## 3. Idiomatic C++23 Features
- Prefer `constexpr` and `consteval` for compile-time calculation.
- Utilize `<ranges>` and standard algorithms over nested index loops.
- Use structured bindings for tuple and pair unpacks.
