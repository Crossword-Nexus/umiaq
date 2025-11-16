use crate::bindings::Bindings;
use crate::solver::{solve_equation, SolverError};
use crate::entry_list::EntryList;
use crate::log::init_logger;
use wasm_bindgen::prelude::*;

use serde_wasm_bindgen::to_value;

/// Structured error information for JavaScript consumers
#[derive(serde::Serialize)]
struct WasmError {
    /// Error code (e.g., "E001", "S002")
    code: String,
    /// Display message
    message: String,
    /// Short description of error type
    description: String,
    /// Detailed explanation
    details: String,
    /// Optional helpful suggestion
    #[serde(skip_serializing_if = "Option::is_none")]
    help: Option<String>,
}

impl From<SolverError> for WasmError {
    fn from(e: SolverError) -> Self {
        // For ParseFailure, extract the nested ParseError details
        match &e {
            SolverError::ParseFailure(pe) => WasmError {
                code: pe.code().to_string(),
                message: pe.to_string(),
                description: pe.description().to_string(),
                details: pe.details().to_string(),
                help: pe.help().map(|s| s.to_string()),
            },
            _ => WasmError {
                code: e.code().to_string(),
                message: e.to_string(),
                description: e.description().to_string(),
                details: e.details().to_string(),
                help: e.help().map(|s| s.to_string()),
            },
        }
    }
}

impl From<WasmError> for JsValue {
    fn from(e: WasmError) -> Self {
        // Format a comprehensive error message
        let mut msg = format!("Error {}: {}", e.code, e.message);

        if !e.details.is_empty() {
            msg.push_str(&format!("\n\n{}", e.details));
        }

        if let Some(help) = e.help {
            msg.push_str(&format!("\n\nSuggestion: {}", help));
        }

        // Create a JavaScript Error object with the formatted message
        js_sys::Error::new(&msg).into()
    }
}

/// Validate all internal regex patterns compile successfully.
///
/// Forces LazyLock initialization of all static regexes so any compilation
/// errors occur at startup rather than on first user query. If a regex pattern
/// is invalid, this will panic with a clear error message.
///
/// ## IMPORTANT: Adding a new regex?
/// If you add a new `LazyLock<Regex>` anywhere in the codebase, you MUST add it here.
/// Otherwise it won't be validated at startup and could crash on first user query.
/// See: `tests::test_all_regexes_validated` for a reminder.
fn validate_internal_regexes() {
    // Access each LazyLock regex to force compilation
    let _ = &*crate::patterns::LEN_CMP_RE;
    let _ = &*crate::patterns::NEQ_RE;
    let _ = &*crate::joint_constraints::JOINT_LEN_RE;
    log::debug!("Internal regex patterns validated successfully");
}

/// Initialize Umiaq logging and validation with the specified debug setting.
///
/// # Arguments
/// * `debug_enabled` - If true, use Debug log level; if false, use Info log level
///
/// This function must be called from JavaScript after the WASM module loads.
#[wasm_bindgen]
pub fn initialize(debug_enabled: bool) {
    // 1. Set up panic hook
    console_error_panic_hook::set_once();

    // 2. Validate internal regexes early
    validate_internal_regexes();

    // 3. Initialize logging with the provided debug setting
    init_logger(debug_enabled);

    log::info!("WASM module initialized");
    if !debug_enabled {
        log::info!("Debug logging disabled");
    }
}

// Pull just the bound entry ("*") out of a Bindings
fn binding_to_entry(b: &Bindings) -> Option<String> {
    b.get_entry().map(|rc| rc.to_string())
}

#[derive(serde::Serialize)]
struct WasmSolveResult {
    solutions: Vec<Vec<String>>,
    status: String,
    readable_equation_context: String,
}

/// JS entry: (input: string, entry_list: string[], num_results_requested: number)
/// returns Array<Array<string>> — only the bound entries
#[wasm_bindgen]
pub fn solve_equation_wasm(
    input: &str,
    entry_list: JsValue,
    num_results_requested: usize,
) -> Result<JsValue, JsValue> {
    // entry_list: string[] -> Vec<String>
    let entries: Vec<String> = serde_wasm_bindgen::from_value(entry_list)
        .map_err(|e| {
            // Create a structured error for deserialization failures
            WasmError {
                code: "WASM001".to_string(),
                message: format!("entry_list must be string[]: {e}"),
                description: "Invalid entry-list format".to_string(),
                details: "The entry_list parameter must be a JavaScript array of strings.".to_string(),
                help: Some("Ensure you're passing a valid string array, e.g., ['cat', 'dog', 'fish']".to_string()),
            }
        })?;
    // Borrow as &[&str] for the solver
    let refs: Vec<&str> = entries.iter().map(|s| s.as_str()).collect();

    let result = solve_equation(input, &refs, num_results_requested)
        .map_err(|e| WasmError::from(e))?;

    let status = match result.status {
        crate::solver::SolveStatus::FoundEnough => "found_enough".to_string(),
        crate::solver::SolveStatus::EntryListExhausted => "entry_list_exhausted".to_string(),
        crate::solver::SolveStatus::TimedOut { .. } => "timed_out".to_string(),
    };

    let wasm_result = WasmSolveResult {
        solutions: result
            .solutions
            .iter()
            .map(|row| row.iter().filter_map(binding_to_entry).collect())
            .collect(),
        status,
        readable_equation_context: result.readable_equation_context,
    };

    serde_wasm_bindgen::to_value(&wasm_result)
        .map_err(|e| {
            WasmError {
                code: "WASM002".to_string(),
                message: format!("serialization failed: {e}"),
                description: "Failed to serialize result".to_string(),
                details: "The solver result could not be converted to JavaScript format.".to_string(),
                help: Some("This is an internal error. Please report this issue.".to_string()),
            }.into()
        })
}

/// Parse a newline-separated entry list string into an `EntryList`.
///
/// Each line of the input should be in the `entry;score` format.
/// Entries with a score below `min_score` are filtered out.
/// Returns the surviving entries as a `JsValue` array of strings,
/// suitable for consumption in JavaScript.
///
/// # Errors
/// Returns a `JsValue` error if parsing fails (e.g. malformed input).
#[wasm_bindgen]
pub fn parse_entry_list(text: &str, min_score: i32) -> Result<JsValue, JsValue> {
    let entry_list = EntryList::parse_from_str(text, min_score);
    to_value(&entry_list.entries)
        .map_err(|e| {
            WasmError {
                code: "WASM003".to_string(),
                message: format!("serialization failed: {e}"),
                description: "Failed to serialize entry list".to_string(),
                details: "The entry list could not be converted to JavaScript format.".to_string(),
                help: Some("This is an internal error. Please report this issue.".to_string()),
            }.into()
        })
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
/// * `entry_list_size` - Number of entries in the entry list
/// * `num_results_requested` - How many results were requested
///
/// # Returns
/// A formatted string containing all debug information
#[wasm_bindgen]
pub fn get_debug_info(
    input_pattern: &str,
    error_message: &str,
    entry_list_size: usize,
    num_results_requested: usize,
) -> String {
    use std::fmt::Write;
    let mut report = String::new();

    // NB: writing to a String never fails (infallible operation)
    // we use `let _ =` to explicitly ignore the Result without panicking
    let _ = writeln!(&mut report, "=== UMIAQ DEBUG REPORT ===");
    let _ = writeln!(&mut report, "Version: {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(&mut report, "Generated: {}", js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_else(|| "unknown".to_string()));
    let _ = writeln!(&mut report);

    let _ = writeln!(&mut report, "## Error");
    let _ = writeln!(&mut report, "{}", error_message);
    let _ = writeln!(&mut report);

    let _ = writeln!(&mut report, "## Input");
    let _ = writeln!(&mut report, "Pattern: {}", input_pattern);
    let _ = writeln!(&mut report, "Entry List Size: {}", entry_list_size);
    let _ = writeln!(&mut report, "Results Requested: {}", num_results_requested);
    let _ = writeln!(&mut report);

    let _ = writeln!(&mut report, "## Environment");
    if let Some(window) = web_sys::window() {
        if let Ok(user_agent) = window.navigator().user_agent() {
            let _ = writeln!(&mut report, "User Agent: {}", user_agent);
        }
        let _ = writeln!(&mut report, "Location: {}", window.location().href().unwrap_or_else(|_| "unknown".to_string()));
    }
    let _ = writeln!(&mut report);

    let _ = writeln!(&mut report, "## Instructions");
    let _ = writeln!(&mut report, "Please copy this entire report and paste it when reporting the issue.");
    let _ = writeln!(&mut report, "GitHub Issues: https://github.com/Crossword-Nexus/umiaq-rust/issues");
    let _ = writeln!(&mut report);

    let _ = writeln!(&mut report, "=== END DEBUG REPORT ===");

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_debug_info_structure() {
        let report = get_debug_info("AB;BA", "solver error: Parse error", 1000, 100);

        let lines: Vec<&str> = report.lines().collect();

        // Verify exact structure
        assert_eq!(lines[0], "=== UMIAQ DEBUG REPORT ===");
        assert_eq!(lines[1], format!("Version: {}", env!("CARGO_PKG_VERSION")));
        assert!(lines[2].starts_with("Generated: ")); // Dynamic timestamp
        assert_eq!(lines[3], "");
        assert_eq!(lines[4], "## Error");
        assert_eq!(lines[5], "solver error: Parse error");
        assert_eq!(lines[6], "");
        assert_eq!(lines[7], "## Input");
        assert_eq!(lines[8], "Pattern: AB;BA");
        assert_eq!(lines[9], "Entry List Size: 1000");
        assert_eq!(lines[10], "Results Requested: 100");
        assert_eq!(lines[11], "");
        assert_eq!(lines[12], "## Environment");
        // lines[13] and [14] are User Agent and Location (dynamic)
        // Find the Instructions section
        let instructions_idx = lines.iter().position(|&l| l == "## Instructions").unwrap();
        assert_eq!(lines[instructions_idx], "## Instructions");
        assert_eq!(lines[instructions_idx + 1], "Please copy this entire report and paste it when reporting the issue.");
        assert_eq!(lines[instructions_idx + 2], "GitHub Issues: https://github.com/Crossword-Nexus/umiaq-rust/issues");
        assert_eq!(lines[instructions_idx + 3], "");
        assert_eq!(lines[instructions_idx + 4], "=== END DEBUG REPORT ===");
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_debug_info_multiline_error() {
        let error_msg = "solver error: Parse error\nLine 2 of error\nLine 3 of error";
        let report = get_debug_info("AB", error_msg, 500, 50);

        let lines: Vec<&str> = report.lines().collect();

        // Find error section
        let error_idx = lines.iter().position(|&l| l == "## Error").unwrap();
        assert_eq!(lines[error_idx + 1], "solver error: Parse error");
        assert_eq!(lines[error_idx + 2], "Line 2 of error");
        assert_eq!(lines[error_idx + 3], "Line 3 of error");
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_debug_info_special_chars_in_pattern() {
        let pattern = "A;B;|A|>5;!=AB;[abc]";
        let report = get_debug_info(pattern, "test error", 2000, 200);

        let lines: Vec<&str> = report.lines().collect();

        // Find input section and verify exact pattern
        let input_idx = lines.iter().position(|&l| l == "## Input").unwrap();
        assert_eq!(lines[input_idx + 1], "Pattern: A;B;|A|>5;!=AB;[abc]");
        assert_eq!(lines[input_idx + 2], "Entry List Size: 2000");
        assert_eq!(lines[input_idx + 3], "Results Requested: 200");
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_debug_info_empty_pattern() {
        let report = get_debug_info("", "Empty pattern error", 0, 1);

        let lines: Vec<&str> = report.lines().collect();

        let input_idx = lines.iter().position(|&l| l == "## Input").unwrap();
        assert_eq!(lines[input_idx + 1], "Pattern: ");
        assert_eq!(lines[input_idx + 2], "Entry List Size: 0");
        assert_eq!(lines[input_idx + 3], "Results Requested: 1");
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_debug_info_large_numbers() {
        let report = get_debug_info("A", "error", 999999, 88888);

        let lines: Vec<&str> = report.lines().collect();

        let input_idx = lines.iter().position(|&l| l == "## Input").unwrap();
        assert_eq!(lines[input_idx + 2], "Entry List Size: 999999");
        assert_eq!(lines[input_idx + 3], "Results Requested: 88888");
    }

    /// Ensure all LazyLock<Regex> statics are validated at startup. (fail fast)
    ///
    /// This test documents which regexes exist and serves as a reminder to update
    /// `validate_internal_regexes()` when adding new regex patterns.
    ///
    /// **If a new `LazyLock<Regex>` is added anywhere in the codebase:**
    /// 1. Add it to `validate_internal_regexes()` in this file
    /// 2. Update this test to include the new regex
    #[test]
    fn test_all_regexes_validated() {
        // this test documents all LazyLock<Regex> statics that should be validated
        // at WASM startup in validate_internal_regexes()

        // current regexes (as of this writing):
        // 1. crate::patterns::LEN_CMP_RE
        // 2. crate::patterns::NEQ_RE
        // 3. crate::joint_constraints::JOINT_LEN_RE

        // if a new regex was added and this test fails, make sure to:
        // - add it to validate_internal_regexes()
        // - update this list

        // force validation to ensure they all compile
        validate_internal_regexes();

        // test passes if no panic occurred above
        assert!(true, "All regexes validated successfully");
    }
}
