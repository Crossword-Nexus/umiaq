//! Error types for parsing operations with error codes and helpful messages.
//!
//! # Error Codes
//!
//! Each error variant has a unique code (E001-E016) for documentation lookup:
//!
//! - E001: `ParseFailure` (Generic parse failure)
//! - E002: `RegexError` (Invalid regex pattern)
//! - E003: `EmptyForm` (Empty form string)
//! - E004: `InvalidLengthRange` (Invalid length range format)
//! - E005: `InvalidComplexConstraint` (Invalid complex constraint)
//! - E006: `InvalidInput` (Invalid input)
//! - E007: `ParseIntError` (Integer parsing error)
//! - E008: `ContradictoryBounds` (Contradictory length bounds)
//! - E009: `InvalidCharsetRange` (Invalid charset range)
//! - E010: `DanglingCharsetDash` (Dangling '-' in charset)
//! - E011: `ConflictingConstraint` (Conflicting variable constraints)
//! - E012: `ClauseParseError` (Parse error in clause (wraps another error))
//! - E013: `InvalidVariableName` (Variable name not A-Z)
//! - E014: `InvalidLowercaseChar` (Non-lowercase character)
//! - E015: `InvalidAnagramChars` (Invalid characters in anagram)
//! - E016: `NomError` (Low-level nom parser error)
//!
//! # Examples
//!
//! ## Basic Error Handling
//!
//! ```
//! use umiaq::errors::ParseError;
//!
//! fn parse_something(input: &str) -> Result<(), Box<ParseError>> {
//!     if input.is_empty() {
//!         return Err(Box::new(ParseError::EmptyForm));
//!     }
//!     Ok(())
//! }
//!
//! match parse_something("") {
//!     Err(e) => {
//!         println!("Error: {}", e);
//!         println!("Code: {}", e.code());
//!         if let Some(help) = e.help() {
//!             println!("Help: {}", help);
//!         }
//!     }
//!     Ok(_) => println!("Success"),
//! }
//! ```
//!
//! ## Error Wrapping with Context
//!
//! ```
//! use umiaq::errors::ParseError;
//!
//! fn parse_low_level(c: char) -> Result<usize, Box<ParseError>> {
//!     if !c.is_ascii_lowercase() {
//!         return Err(Box::new(ParseError::InvalidLowercaseChar { invalid_char: c }));
//!     }
//!     Ok(c as usize - 'a' as usize)
//! }
//!
//! fn parse_high_level(word: &str) -> Result<Vec<usize>, Box<ParseError>> {
//!     word.chars().map(|c| {
//!         parse_low_level(c).map_err(|e| {
//!             // Wrap with additional context
//!             if let ParseError::InvalidLowercaseChar { invalid_char } = *e {
//!                 Box::new(ParseError::InvalidAnagramChars {
//!                     anagram: word.to_string(),
//!                     invalid_char
//!                 })
//!             } else {
//!                 e
//!             }
//!         })
//!     }).collect()
//! }
//! ```

use nom::error::{ErrorKind, ParseError as NomParseError};
use std::num::ParseIntError;
use std::io;


/// Custom error type for parsing operations
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Form parsing failed: \"{s}\"")]
    ParseFailure { s: String },

    #[error("Invalid regex pattern: {0}")]
    RegexError(#[from] fancy_regex::Error),

    #[error("Empty form string")]
    EmptyForm,

    #[error("Invalid length range: \"{input}\"")]
    InvalidLengthRange { input: String },

    #[error("{str}")]
    InvalidComplexConstraint { str: String },

    #[error("Invalid input: {str}")]
    InvalidInput { str: String },

    #[error("int-parsing error: {0}")]
    ParseIntError(#[from] ParseIntError),

    #[error("contradictory bounds: min={min}, max={max}")]
    ContradictoryBounds {
        min: usize,
        max: usize,
    },

    #[error("Invalid range in charset: {0}-{1}")]
    InvalidCharsetRange(char, char),

    #[error("Dangling '-' at end of charset")]
    DanglingCharsetDash,

    #[error("Conflicting constraints for {var_char} ({older} / {newer})")]
    ConflictingConstraint { var_char: char, older: String, newer: String },

    #[error("Parse error in clause '{clause}': {source}")]
    ClauseParseError {
        clause: String,
        #[source]
        source: Box<ParseError>,
    },

    #[error("Invalid variable name '{var}' (must be A-Z)")]
    InvalidVariableName { var: String },

    #[error("Invalid character '{invalid_char}' (only lowercase a-z allowed)")]
    InvalidLowercaseChar { invalid_char: char },

    #[error("Anagram constraint \"{anagram}\" contains invalid character '{invalid_char}' (only a-z allowed)")]
    InvalidAnagramChars { anagram: String, invalid_char: char },

    // // TODO use(?) (e.g., when detecting patterns with excessive nesting, backtracking potential)
    // #[error("Pattern too complex: {reason}")]
    // PatternComplexity { reason: String },

    // nom parser error (lowest level)
    #[error("nom parser error: {0:?}")]
    NomError(ErrorKind),
}

impl From<ParseError> for io::Error {
    fn from(pe: ParseError) -> Self {
        // String version is the least fragile (no Send/Sync bounds issues)
        io::Error::new(io::ErrorKind::InvalidInput, pe.to_string())
    }
}

impl From<ParseIntError> for Box<ParseError> {
    fn from(pie: ParseIntError) -> Self {
        Box::new(ParseError::ParseIntError(pie))
    }
}

impl From<Box<fancy_regex::Error>> for Box<ParseError> {
    fn from(e: Box<fancy_regex::Error>) -> Self {
        Box::new(ParseError::RegexError(*e))
    }
}

impl<'a> NomParseError<&'a str> for Box<ParseError> {
    fn from_error_kind(_input: &'a str, kind: ErrorKind) -> Self {
        Box::new(ParseError::NomError(kind))
    }

    fn append(_input: &'a str, _kind: ErrorKind, other: Self) -> Self {
        // usually just return the existing error unchanged
        other
    }
}

impl ParseError {
    /// Returns the error code for this error variant
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            ParseError::ParseFailure { .. } => "E001",
            ParseError::RegexError(_) => "E002",
            ParseError::EmptyForm => "E003",
            ParseError::InvalidLengthRange { .. } => "E004",
            ParseError::InvalidComplexConstraint { .. } => "E005",
            ParseError::InvalidInput { .. } => "E006",
            ParseError::ParseIntError(_) => "E007",
            ParseError::ContradictoryBounds { .. } => "E008",
            ParseError::InvalidCharsetRange(..) => "E009",
            ParseError::DanglingCharsetDash => "E010",
            ParseError::ConflictingConstraint { .. } => "E011",
            ParseError::ClauseParseError { .. } => "E012",
            ParseError::InvalidVariableName { .. } => "E013",
            ParseError::InvalidLowercaseChar { .. } => "E014",
            ParseError::InvalidAnagramChars { .. } => "E015",
            ParseError::NomError(_) => "E016",
        }
    }

    /// Returns a helpful suggestion or example for this error
    #[must_use]
    pub fn help(&self) -> Option<&'static str> {
        match self {
            ParseError::EmptyForm => Some("Example: Use 'A*B' or '*cat*' instead of empty string"),
            ParseError::InvalidLengthRange { .. } => Some("Expected format: N-M where N ≤ M (e.g., '3-5' or '1-10')"),
            ParseError::ContradictoryBounds { .. } => Some("The minimum length cannot exceed the maximum length"),
            ParseError::InvalidCharsetRange(..) => Some("In a charset range, the first character must come before the second (e.g., 'a-z' not 'z-a')"),
            ParseError::DanglingCharsetDash => Some("Remove the trailing '-' or complete the range (e.g., '[abc]' or '[a-c]')"),
            ParseError::ConflictingConstraint { .. } => Some("Variable has incompatible constraints that cannot both be satisfied"),
            ParseError::InvalidVariableName { .. } => Some("Variable names must be single uppercase letters A-Z"),
            ParseError::InvalidLowercaseChar { .. } => Some("Only lowercase letters a-z are allowed"),
            ParseError::InvalidAnagramChars { .. } => Some("Anagrams must contain only lowercase letters a-z"),
            _ => None,
        }
    }

    /// Formats the error with code and optional help text
    #[must_use]
    pub fn display_detailed(&self) -> String {
        format_error_with_code_and_help(&self.to_string(), self.code(), self.help())
    }
}

/// Helper function to format error messages with code and optional help text
pub(crate) fn format_error_with_code_and_help(base_msg: &str, code: &str, help: Option<&str>) -> String {
    if let Some(help_text) = help {
        format!("{base_msg} ({code})\n{help_text}")
    } else {
        format!("{base_msg} ({code})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes_and_help() {
        let err = ParseError::EmptyForm;
        assert_eq!(err.code(), "E003");
        assert!(err.help().is_some());
        let detailed = err.display_detailed();
        assert!(detailed.contains("E003"));
        assert!(detailed.contains("Example"));
    }

    #[test]
    fn test_contradictory_bounds_help() {
        let err = ParseError::ContradictoryBounds { min: 5, max: 3 };
        assert_eq!(err.code(), "E008");
        let detailed = err.display_detailed();
        assert!(detailed.contains("minimum length cannot exceed"));
    }

    /// Test that all `ParseError` variants have unique error codes
    #[test]
    fn test_all_error_codes_are_unique() {
        let mut codes = std::collections::HashSet::new();

        // Sample one of each variant
        let errors: Vec<ParseError> = vec![
            ParseError::ParseFailure { s: "test".to_string() },
            ParseError::EmptyForm,
            ParseError::InvalidInput { str: "test".to_string() },
            ParseError::InvalidVariableName { var: "1".to_string() },
            ParseError::ContradictoryBounds { min: 5, max: 3 },
            ParseError::InvalidLengthRange { input: "bad".to_string() },
            ParseError::InvalidComplexConstraint { str: "bad".to_string() },
            ParseError::InvalidCharsetRange('a', 'z'),
            ParseError::DanglingCharsetDash,
            ParseError::ConflictingConstraint { var_char: 'A', older: "old".to_string(), newer: "new".to_string() },
            ParseError::InvalidLowercaseChar { invalid_char: 'X' },
            ParseError::InvalidAnagramChars { anagram: "xyz".to_string(), invalid_char: 'X' },
        ];

        for err in errors {
            let code = err.code();
            assert!(
                code.starts_with('E'),
                "Error code '{}' should start with 'E'",
                code
            );
            assert!(
                codes.insert(code),
                "Duplicate error code found: {}",
                code
            );
        }

        // Verify we tested at least 9 variants
        assert!(codes.len() >= 9, "Should have at least 9 unique error codes");
    }

    /// Test that all error codes follow the format E0XX
    #[test]
    fn test_error_code_format() {
        let errors: Vec<ParseError> = vec![
            ParseError::EmptyForm,
            ParseError::ContradictoryBounds { min: 5, max: 3 },
            ParseError::InvalidInput { str: "test".to_string() },
        ];

        for err in errors {
            let code = err.code();
            assert_eq!(code.len(), 4, "Error code '{}' should be 4 characters (E0XX)", code);
            assert!(
                code.starts_with("E0"),
                "Error code '{}' should start with 'E0'",
                code
            );
            let num_part = &code[1..];
            assert!(
                num_part.parse::<u16>().is_ok(),
                "Error code '{}' should end with a number",
                code
            );
        }
    }

    /// Test that all errors have helpful help text
    #[test]
    fn test_all_errors_have_helpful_messages() {
        let errors: Vec<ParseError> = vec![
            ParseError::EmptyForm,
            ParseError::ContradictoryBounds { min: 5, max: 3 },
            ParseError::InvalidInput { str: "test".to_string() },
            ParseError::InvalidVariableName { var: "1".to_string() },
            ParseError::InvalidLengthRange { input: "bad".to_string() },
        ];

        for err in errors {
            let help = err.help();
            if let Some(help_text) = help {
                assert!(
                    help_text.len() > 10,
                    "Help text for {:?} should be substantial",
                    err
                );
                // Help text should not just repeat the error message
                let err_msg = err.to_string();
                assert_ne!(help_text, err_msg, "Help text should provide additional information beyond error message");
            }
            // Not all errors need help text, so we don't assert help.is_some()
        }
    }

    /// Test that display_detailed properly formats errors
    #[test]
    fn test_display_detailed_includes_code_and_help() {
        let err = ParseError::EmptyForm;
        let detailed = err.display_detailed();

        // should include code
        assert!(
            detailed.contains(err.code()),
            "Detailed display should include error code"
        );

        // should include base message
        let base_msg = err.to_string();
        assert!(
            detailed.contains(&base_msg),
            "Detailed display should include base error message"
        );

        // if there's help text, it should be included
        if let Some(help) = err.help() {
            assert!(
                detailed.contains(help),
                "Detailed display should include help text when available"
            );
        }
    }

    /// Test that error messages are useful
    #[test]
    fn test_error_messages_are_actionable() {
        let err = ParseError::ContradictoryBounds { min: 5, max: 3 };
        let detailed = err.display_detailed();

        // should explain what went wrong
        assert!(
            detailed.contains("minimum") || detailed.contains("min"),
            "Error should mention the problematic constraint"
        );

        // should include the actual values
        assert!(
            detailed.contains('5') && detailed.contains('3'),
            "Error should include the actual conflicting values"
        );
    }

    /// Test error chain construction for `ClauseParseError`
    #[test]
    fn test_clause_parse_error_chain() {
        let source = ParseError::InvalidInput { str: "bad".to_string() };
        let err = ParseError::ClauseParseError {
            clause: "bad".to_string(),
            source: Box::new(source),
        };

        let detailed = err.display_detailed();

        // should show the clause
        assert!(
            detailed.contains("bad"),
            "ClauseParseError should show the problematic clause"
        );

        // should show error codes from both levels
        assert!(
            detailed.contains(err.code()),
            "Should show outer error code"
        );
    }
}
