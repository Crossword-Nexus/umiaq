//! Umiaq is a pattern-matching solver for crossword puzzles and similar word games.
//!
//! # Core Functionality
//!
//! The main entry point is [`solver::solve_equation`], which takes a pattern string and
//! a word list, returning matching solutions.
//!
//! # Error Handling
//!
//! Umiaq uses structured error types with error codes and helpful messages:
//!
//! - [`solver::SolverError`] (Top-level errors (S001-S003))
//! - [`errors::ParseError`] (Parsing errors (E001-E016))
//!
//! ## Example: Basic Usage
//!
//! ```
//! use umiaq::solver;
//!
//! let words = vec!["cat", "dog", "catalog"];
//! let result = solver::solve_equation("A*B", &words, 10)?;
//!
//! for solution in result.solutions {
//!     println!("{}", solver::solution_to_string(&solution)?);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Example: Error Handling with Detailed Messages
//!
//! ```
//! use umiaq::solver::{self, SolverError};
//!
//! let words = vec!["test"];
//! let result = solver::solve_equation("", &words, 10);
//!
//! match result {
//!     Ok(r) => println!("Found {} solutions", r.solutions.len()),
//!     Err(e) => {
//!         // Display with error code and help text
//!         eprintln!("Error: {}", e.display_detailed());
//!         // Error code for documentation lookup
//!         eprintln!("Code: {}", e.code());
//!     }
//! }
//! ```
//!
//! ## Example: Handling Specific Error Types
//!
//! ```
//! use umiaq::solver::{self, SolverError};
//! use umiaq::errors::ParseError;
//!
//! let words = vec!["test"];
//! match solver::solve_equation("BAD(PATTERN", &words, 10) {
//!     Err(SolverError::ParseFailure(parse_err)) => {
//!         // Access the specific ParseError
//!         match &*parse_err {
//!             ParseError::InvalidInput { str } => {
//!                 println!("Invalid input: {}", str);
//!             }
//!             ParseError::ClauseParseError { clause, source } => {
//!                 println!("Error in clause '{}': {}", clause, source);
//!             }
//!             _ => println!("Other parse error: {}", parse_err),
//!         }
//!     }
//!     Err(e) => println!("Solver error: {}", e),
//!     Ok(_) => println!("Success"),
//! }
//! ```

// Reusable library API — visible to both CLI and WASM builds
pub mod bindings;
pub mod constraints;
pub mod errors;
mod interner;
mod joint_constraints;
pub mod parser;
pub mod patterns;
mod scan_hints;
pub mod solver;
pub mod umiaq_char;
pub mod word_list;
pub mod complex_constraints;

// Compile the wasm glue only when targeting wasm32.
mod comparison_operator;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

