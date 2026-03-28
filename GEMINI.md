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
- **`src/solver.rs`**: Core engine. Uses adaptive batching (`scan_batch`) and backtracking (`recursive_join`) with variable bucketing to join patterns. Features a 30s `TIME_BUDGET`.
- **`src/parser/`**:
    - `form.rs`: Defines `FormPart` (Lit, Var, RevVar, Dot, Star, Vowel `@`, Consonant `#`, Charset `[]`, Anagram `/`).
    - `matcher.rs`: Backtracking matcher for single entries.
    - `prefilter.rs`: Generates optimized regexes with lookaheads for variable constraints.
- **`src/patterns.rs` & `src/constraints.rs`**: `get_ordered_patterns` uses a greedy heuristic (`constraint_score`) to minimize search space (Literals=3, Classes=1).
- **`src/scan_hints.rs`**: Analyzes pattern structure to compute length bounds used in `scan_batch`.
- **`src/umiaq_char.rs`**: Character definitions. Note: 'y' is considered a vowel.

## Search Primitives & Optimizations
- **Variable Bucketing**: Candidates for pattern `i` are grouped by variables shared with patterns `0..i-1`.
- **Regex Prefiltering**: Each pattern has a regex prefilter; if a variable has a form constraint (e.g., `A;A=cat*`), it is inlined as a lookahead.
- **Deterministic Fast Path**: If a pattern's variables are fully bound, it materializes and checks the dictionary directly.
- **Wildcards**: `@` (vowels), `#` (consonants), `.` (any), `*` (zero-or-more).
- **Operators**: `/` (anagram), `~` (reverse variable).

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
