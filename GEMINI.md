# Umiaq: Gemini CLI Context

This file provides context for Gemini CLI to understand the Umiaq project's architecture, technology stack, and development workflows.

## Project Overview
Umiaq is an open-source solver and entry-pattern matching tool designed for crossword constructors and wordplay enthusiasts. It allows searching large entry lists (dictionaries) using expressive patterns, variables, and constraints.

## Tech Stack
- **Language**: Rust (Edition 2024)
- **Parsing**: [nom](https://github.com/rust-bakery/nom) for pattern/constraint parsing.
- **WASM**: [wasm-pack](https://github.com/rustwasm/wasm-pack) and `wasm-bindgen` for the web interface.
- **Regex**: [fancy-regex](https://github.com/fancy-regex/fancy-regex) for advanced pattern matching.
- **CLI**: [clap](https://github.com/clap-rs/clap) for command-line argument parsing.

## Key Architectural Components
- **`src/solver.rs`**: The core engine that coordinates pattern matching and constraint satisfaction.
- **`src/parser/`**: Contains the logic for parsing the domain-specific language (DSL) into internal representations.
- **`src/patterns.rs` & `src/constraints.rs`**: Define the matching logic for variables, literals, wildcards, and various constraint types.
- **`src/wasm.rs`**: The WebAssembly interface, providing bindings for the JavaScript frontend.
- **`src/entry_list.rs`**: Handles loading and scoring of word lists (typically `.dict` files).
- **`src/scan_hints.rs`**: Computes length bounds and pre-filters for patterns to optimize search performance.
- **`src/errors.rs`**: Comprehensive error system with unique error codes (E001-E022, S001-S003).

## Common Commands
- **Run Tests**: `cargo test`
- **Build WASM**: `wasm-pack build --release --target web --out-dir web/pkg`
- **Run Local Benchmark**: `cargo run --bin bench_local --release -- -r 1 -p 0`
- **Run CLI**: `cargo run -- "A*B;A=cat"`

## Coding Standards & Conventions
- **Error Handling**: Use the structured `ParseError` and `SolverError` types. Each error must have a unique code and helpful message.
- **Documentation**: All public modules and functions should have doc comments with examples.
- **Performance**: Be mindful of pattern ordering; the solver's efficiency depends on minimizing the search space through variable bucketing.
- **Testing**: New features should include integration tests in `tests/integration_tests.rs` and unit tests in the relevant source files.
