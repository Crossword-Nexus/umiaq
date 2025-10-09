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
