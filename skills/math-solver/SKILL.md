---
name: math-solver
version: 1.0.0
description: Step-by-step mathematical reasoning and verified numerical evaluation
category: math
author: QorVix
tags: ["math", "calculator", "algebra", "arithmetic"]
required_tools: ["calculator"]
---

# Math Reasoning Playbook

When solving mathematical, algebraic, or numerical calculations:

## 1. Problem Decomposition
- State given variables, constants, and target unknowns explicitly.
- State algebraic equations and principles to apply.

## 2. Step-by-Step Derivation
- Write out intermediate derivations before computing final values.
- Maintain symbolic notation until numerical substitution.

## 3. Tool Verification
- Always execute numerical calculations through the `calculator` tool:
  `Action: calculator`
  `Action Input: {"expression": "..."}`

## 4. Dimensional & Sanity Check
- Verify units and physical dimensions if applicable.
- Confirm boundary values and sign consistency.
