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
    pub fn display_detailed(&self) -> String {
        format_error_with_code_and_help(&self.to_string(), self.code(), self.help())
    }
}

/// Helper function to format error messages with code and optional help text (DRY)
pub(crate) fn format_error_with_code_and_help(base_msg: &str, code: &str, help: Option<&str>) -> String {
    if let Some(help_text) = help {
        format!("{} ({})\n{}", base_msg, code, help_text)
    } else {
        format!("{} ({})", base_msg, code)
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
}
