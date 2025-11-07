use crate::bindings::Bindings;
use crate::errors::ParseError;
use crate::solver::solve_equation;
use crate::word_list::WordList;
use crate::log::init_logger;
use wasm_bindgen::prelude::*;

use serde_wasm_bindgen::to_value;

/// Implement `Box<ParseError>` for `JsValue`s
impl From<Box<ParseError>> for JsValue {
    fn from(e: Box<ParseError>) -> JsValue { JsValue::from_str(format!("[parse error] {}", *e).as_str()) }
}

#[wasm_bindgen(start)]
fn init() {
    // 1. Set up panic hook
    console_error_panic_hook::set_once();

    // 2. Initialize logging — always debug-level in WASM
    init_logger(true);

    log::info!("WASM module initialized");
}

// Pull just the bound word ("*") out of a Bindings
fn binding_to_word(b: &Bindings) -> Option<String> {
    b.get_word().map(|rc| rc.to_string())
}

#[derive(serde::Serialize)]
struct WasmSolveResult {
    solutions: Vec<Vec<String>>,
    status: String,
    readable_equation_context: String,
}

/// JS entry: (input: string, word_list: string[], num_results_requested: number)
/// returns Array<Array<string>> — only the bound words
// TODO remove unnecessary type annotations
#[wasm_bindgen]
pub fn solve_equation_wasm(
    input: &str,
    word_list: JsValue,
    num_results_requested: usize,
) -> Result<JsValue, JsValue> {
    // word_list: string[] -> Vec<String>
    let words: Vec<String> = serde_wasm_bindgen::from_value(word_list)
        .map_err(|e| JsValue::from_str(&format!("word_list must be string[]: {e}")))?;
    // Borrow as &[&str] for the solver
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    let result = solve_equation(input, &refs, num_results_requested)
        .map_err(|e| JsValue::from_str(&format!("solver error: {e}")))?;

    let status = match result.status {
        crate::solver::SolveStatus::FoundEnough => "found_enough".to_string(),
        crate::solver::SolveStatus::WordListExhausted => "word_list_exhausted".to_string(),
        crate::solver::SolveStatus::TimedOut { .. } => "timed_out".to_string(),
    };

    let wasm_result = WasmSolveResult {
        solutions: result
            .solutions
            .iter()
            .map(|row| row.iter().filter_map(binding_to_word).collect())
            .collect(),
        status,
        readable_equation_context: result.readable_equation_context,
    };

    serde_wasm_bindgen::to_value(&wasm_result)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

/// Parse a newline-separated word list string into a `WordList`.
///
/// Each line of the input should be in the `word;score` format.
/// Words with a score below `min_score` are filtered out.
/// Returns the surviving entries as a `JsValue` array of strings,
/// suitable for consumption in JavaScript.
///
/// # Errors
/// Returns a `JsValue` error if parsing fails (e.g. malformed input).
#[wasm_bindgen]
pub fn parse_word_list(text: &str, min_score: i32) -> Result<JsValue, JsValue> {
    let word_list = WordList::parse_from_str(text, min_score);
    to_value(&word_list.entries)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}

/// Generate a debug report for troubleshooting.
///
/// This function creates a formatted debug report that users can copy/paste
/// when reporting issues. It includes the error message, input pattern,
/// configuration details, and environment information.
///
/// # Arguments
/// * `input_pattern` - The equation pattern that was being solved
/// * `error_message` - The error message that was displayed
/// * `word_list_size` - Number of words in the word list
/// * `num_results_requested` - How many results were requested
///
/// # Returns
/// A formatted string containing all debug information
#[wasm_bindgen]
pub fn get_debug_info(
    input_pattern: &str,
    error_message: &str,
    word_list_size: usize,
    num_results_requested: usize,
) -> String {
    use std::fmt::Write;
    let mut report = String::new();

    writeln!(&mut report, "=== UMIAQ DEBUG REPORT ===").unwrap();
    writeln!(&mut report, "Version: {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(&mut report, "Generated: {}", js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_else(|| "unknown".to_string())).unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Error").unwrap();
    writeln!(&mut report, "{}", error_message).unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Input").unwrap();
    writeln!(&mut report, "Pattern: {}", input_pattern).unwrap();
    writeln!(&mut report, "Word List Size: {}", word_list_size).unwrap();
    writeln!(&mut report, "Results Requested: {}", num_results_requested).unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Environment").unwrap();
    if let Some(window) = web_sys::window() {
        if let Ok(user_agent) = window.navigator().user_agent() {
            writeln!(&mut report, "User Agent: {}", user_agent).unwrap();
        }
        writeln!(&mut report, "Location: {}", window.location().href().unwrap_or_else(|_| "unknown".to_string())).unwrap();
    }
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "## Instructions").unwrap();
    writeln!(&mut report, "Please copy this entire report and paste it when reporting the issue.").unwrap();
    writeln!(&mut report, "GitHub Issues: https://github.com/Crossword-Nexus/umiaq-rust/issues").unwrap();
    writeln!(&mut report).unwrap();

    writeln!(&mut report, "=== END DEBUG REPORT ===").unwrap();

    report
}
