// Reusable library API — visible to both CLI and WASM builds
pub mod bindings;
pub mod constraints;
mod errors;
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

