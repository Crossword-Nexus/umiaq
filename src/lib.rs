// Reusable library API — visible to both CLI and WASM builds
pub mod word_list;
pub mod solver;
pub mod bindings;
pub mod parser;
pub mod patterns;
pub mod constraints;
pub mod umiaq_char;
mod joint_constraints;
mod scan_hints;
mod errors;

// Compile the wasm glue only when targeting wasm32.
#[cfg(target_arch = "wasm32")]
pub mod wasm;
mod comparison_operator;
