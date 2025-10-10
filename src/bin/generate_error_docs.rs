//! Generate error code documentation from the source of truth (error enums).
//!
//! This binary reads the error codes, descriptions, details, and help text
//! directly from the `ParseError` and `SolverError` implementations via their
//! `code()`, `description()`, `details()`, and `help()` methods.
//!
//! Run with:
//! ```bash
//! cargo run --bin generate_error_docs > docs/ERROR_CODES.md
//! ```

use umiaq::errors::ParseError;
use umiaq::solver::SolverError;

/// Macro to generate error documentation for any error type
/// with `code()`, `description()`, `details()`, `help()`, and `display_detailed()` methods
macro_rules! generate_error_docs {
    ($errors:expr) => {
        for error in $errors {
            let code = error.code();
            let description = error.description();
            let details = error.details();
            let help = error.help();

            println!("### {}: {}\n", code, description);
            println!("**Details:** {}\n", details);

            if let Some(help_text) = help {
                println!("**How to fix:**");
                println!("```");
                println!("{}", help_text);
                println!("```\n");
            }

            println!("**Example error message:**");
            println!("```");
            println!("{}", error);
            println!("```\n");

            println!("**Detailed format:**");
            println!("```");
            println!("{}", error.display_detailed());
            println!("```\n");

            println!("---\n");
        }
    };
}

/// Helper to create all `ParseError` variants for documentation
fn all_parse_error_variants() -> Vec<ParseError> {
    vec![
        ParseError::ParseFailure { s: "BAD(INPUT".to_string() },
        // RegexError--create by attempting to compile an invalid regex
        ParseError::RegexError(fancy_regex::Regex::new("(?P<invalid").unwrap_err()),
        ParseError::EmptyForm,
        ParseError::InvalidLengthRange { input: "10-5".to_string() },
        ParseError::InvalidComplexConstraint { str: "bad constraint".to_string() },
        ParseError::InvalidInput { str: "bad input".to_string() },
        // ParseIntError--create by parsing invalid integer
        ParseError::ParseIntError("not_a_number".parse::<usize>().unwrap_err()),
        ParseError::ContradictoryBounds { min: 5, max: 3 },
        ParseError::InvalidCharsetRange('z', 'a'),
        ParseError::DanglingCharsetDash,
        ParseError::ConflictingConstraint {
            var_char: 'A',
            older: "|A|=3".to_string(),
            newer: "|A|=5".to_string()
        },
        ParseError::ClauseParseError {
            clause: "BAD(INPUT".to_string(),
            source: Box::new(ParseError::InvalidInput { str: "BAD(INPUT".to_string() })
        },
        ParseError::InvalidVariableName { var: "1".to_string() },
        ParseError::InvalidLowercaseChar { invalid_char: 'X' },
        ParseError::InvalidAnagramChars {
            anagram: "XYZ".to_string(),
            invalid_char: 'X'
        },
        // NomError--use a common error kind
        ParseError::NomError(nom::error::ErrorKind::Tag),
    ]
}

/// Helper to create all `SolverError` variants for documentation
fn all_solver_error_variants() -> Vec<SolverError> {
    vec![
        SolverError::ParseFailure(Box::new(ParseError::EmptyForm)),
        SolverError::NoPatterns,
        SolverError::MaterializationError {
            context: "pattern depth: 2/3, environment: {A: \"test\"}".to_string()
        },
    ]
}

fn main() {
    println!("# Error Code Reference\n");
    println!("**⚠️ This document is auto-generated from the source code. Do not edit manually.**\n");

    println!("## Table of Contents\n");
    println!("- [Solver Errors (S001–S003)](#solver-errors)");
    println!("- [Parse Errors (E001–E016)](#parse-errors)");
    println!("- [How to Use Error Codes](#how-to-use-error-codes)\n");

    generate_solver_error_docs();
    generate_parse_error_docs();

    println!("\n## How to Use Error Codes\n");
    println!("When you see an error like:\n");
    println!("```");
    println!("Error: Empty form string (E003)");
    println!("Example: Use 'A*B' or '*cat*' instead of empty string");
    println!("```\n");
    println!("1. Note the error code (e.g., `E003`)");
    println!("2. Look it up in this document for detailed explanation");
    println!("3. Follow the suggested resolution steps\n");

    println!("## Error Display Formats\n");
    println!("Errors are displayed in two formats:\n");
    println!("### Simple Format");
    println!("```");
    println!("Error: <message>");
    println!("```\n");
    println!("### Detailed Format (via `display_detailed()`)");
    println!("```");
    println!("<message> (<code>)");
    println!("<help text if available>");
    println!("```\n");
}

fn generate_solver_error_docs() {
    println!("## Solver Errors\n");
    println!("Top-level errors from the solver. These wrap lower-level parse errors or indicate solver-specific issues.\n");
    generate_error_docs!(all_solver_error_variants());
}

fn generate_parse_error_docs() {
    println!("## Parse Errors\n");
    println!("Errors that occur when parsing pattern strings, constraints, or equations.\n");
    generate_error_docs!(all_parse_error_variants());
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    /// Captures stdout during documentation generation
    fn generate_docs_to_string() -> String {
        use std::process::{Command, Stdio};

        let output = Command::new("cargo")
            .args(&["run", "--bin", "generate_error_docs"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .expect("Failed to run generate_error_docs");

        String::from_utf8(output.stdout).expect("Invalid UTF-8 in generated docs")
    }

    #[test]
    fn test_generated_docs_match_expected() {
        let generated = generate_docs_to_string();
        let expected = include_str!("../../tests/resources/expected_error_codes.md");

        if generated != expected {
            // Write the diff to a file for easier debugging
            let mut diff_file = std::fs::File::create("tests/resources/error_codes_diff.txt")
                .expect("Failed to create diff file");
            writeln!(diff_file, "=== EXPECTED ===").unwrap();
            writeln!(diff_file, "{}", expected).unwrap();
            writeln!(diff_file, "\n=== GENERATED ===").unwrap();
            writeln!(diff_file, "{}", generated).unwrap();

            panic!(
                "Generated error documentation does not match expected.\n\
                 Diff written to tests/resources/error_codes_diff.txt\n\
                 If the changes are intentional, update the baseline:\n\
                 cargo run --bin generate_error_docs > tests/resources/expected_error_codes.md"
            );
        }
    }
}
