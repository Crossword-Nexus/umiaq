//! Generate error code documentation from the source of truth (error enums).
//!
//! This binary reads the error codes, descriptions, details, and help text
//! directly from the `ParseError` and `SolverError` implementations via their
//! `code()`, `description()`, `details()`, and `help()` methods.
//!
//! Run with:
//! ```bash
//! cargo run --bin generate_error_docs [OUTPUT_FILE]
//! ```
//! Defaults to `docs/ERROR_CODES.md`. Use `-` for stdout.

use umiaq::errors::ParseError;
use umiaq::solver::SolverError;
use std::fs::File;
use std::io::{self, Write, BufWriter};
use std::env;

/// Macro to generate error documentation for any error type
/// with `code()`, `description()`, `details()`, `help()`, and `display_detailed()` methods
macro_rules! generate_error_docs {
    ($errors:expr, $writer:expr) => {
        for error in $errors {
            let code = error.code();
            let description = error.description();
            let details = error.details();
            let help = error.help();

            writeln!($writer, "### {}: {}\n", code, description)?;
            writeln!($writer, "**Details:** {}\n", details)?;

            if let Some(help_text) = help {
                writeln!($writer, "**How to fix:**")?;
                writeln!($writer, "```")?;
                writeln!($writer, "{}", help_text)?;
                writeln!($writer, "```\n")?;
            }

            writeln!($writer, "**Example error message:**")?;
            writeln!($writer, "```")?;
            writeln!($writer, "{}", error)?;
            writeln!($writer, "```\n")?;

            writeln!($writer, "**Detailed format:**")?;
            writeln!($writer, "```")?;
            writeln!($writer, "{}", error.display_detailed())?;
            writeln!($writer, "```\n")?;

            writeln!($writer, "---\n")?;
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
        ParseError::InvalidInput { str: "bad input".to_string(), reason: "bad input".to_string() },
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
            source: Box::new(ParseError::InvalidInput { str: "BAD(INPUT".to_string(), reason: "illegal character '(' in pattern".to_string() })
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

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    let output_path = if args.len() > 1 {
        &args[1]
    } else {
        "docs/ERROR_CODES.md"
    };

    let mut writer: Box<dyn Write> = if output_path == "-" {
        Box::new(BufWriter::new(io::stdout()))
    } else {
        let file = File::create(output_path)?;
        Box::new(BufWriter::new(file))
    };

    writeln!(writer, "# Error Code Reference\n")?;
    writeln!(writer, "**⚠️ This document is auto-generated from the source code. Do not edit manually.**\n")?;

    writeln!(writer, "## Table of Contents\n")?;
    writeln!(writer, "- [Solver Errors (S001–S003)](#solver-errors)")?;
    writeln!(writer, "- [Parse Errors (E001–E016)](#parse-errors)")?;
    writeln!(writer, "- [How to Use Error Codes](#how-to-use-error-codes)\n")?;

    writeln!(writer, "## Solver Errors\n")?;
    writeln!(writer, "Top-level errors from the solver. These wrap lower-level parse errors or indicate solver-specific issues.\n")?;
    generate_error_docs!(all_solver_error_variants(), writer);

    writeln!(writer, "## Parse Errors\n")?;
    writeln!(writer, "Errors that occur when parsing pattern strings, constraints, or equations.\n")?;
    generate_error_docs!(all_parse_error_variants(), writer);

    writeln!(writer, "\n## How to Use Error Codes\n")?;
    writeln!(writer, "When you see an error like:\n")?;
    writeln!(writer, "```")?;
    writeln!(writer, "Error: Empty form string (E003)")?;
    writeln!(writer, "Example: Use 'A*B' or '*cat*' instead of empty string")?;
    writeln!(writer, "```\n")?;
    writeln!(writer, "1. Note the error code (e.g., `E003`)")?;
    writeln!(writer, "2. Look it up in this document for detailed explanation")?;
    writeln!(writer, "3. Follow the suggested resolution steps\n")?;

    writeln!(writer, "## Error Display Formats\n")?;
    writeln!(writer, "Errors are displayed in two formats:\n")?;
    writeln!(writer, "### Simple Format")?;
    writeln!(writer, "```")?;
    writeln!(writer, "Error: <message>")?;
    writeln!(writer, "```\n")?;
    writeln!(writer, "### Detailed Format (via `display_detailed()`)")?;
    writeln!(writer, "```")?;
    writeln!(writer, "<message> (<code>)")?;
    writeln!(writer, "<help text if available>")?;
    writeln!(writer, "```\n")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    /// Captures output during documentation generation
    fn generate_docs_to_string() -> String {
        use std::process::{Command, Stdio};

        let output = Command::new("cargo")
            .args(&["run", "--bin", "generate_error_docs", "--", "-"])
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

        // Normalize line endings for cross-platform comparison
        // TODO: is this the right way to handle this?
        let normalized_generated = generated.replace("\r\n", "\n");
        let normalized_expected = expected.replace("\r\n", "\n");

        if normalized_generated != normalized_expected {
            // Write the diff to a file for easier debugging
            let mut diff_file = std::fs::File::create("tests/resources/error_codes_diff.txt")
                .expect("Failed to create diff file");
            writeln!(diff_file, "=== EXPECTED ===").unwrap();
            writeln!(diff_file, "{}", normalized_expected).unwrap();
            writeln!(diff_file, "\n=== GENERATED ===").unwrap();
            writeln!(diff_file, "{}", normalized_generated).unwrap();

            panic!(
                "Generated error documentation does not match expected.\n\
                 Diff written to tests/resources/error_codes_diff.txt\n\
                 If the changes are intentional, update the baseline:\n\
                 cargo run --bin generate_error_docs tests/resources/expected_error_codes.md"
            );
        }
    }
}
