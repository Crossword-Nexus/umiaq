use crate::errors::ParseError::ParseFailure;
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

    // ... existing variants, e.g., from variables, constraints, etc. ...
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
        ParseFailure { s: pie.to_string() }.into()
    }
}

impl From<Box<fancy_regex::Error>> for Box<ParseError> {
    fn from(e: Box<fancy_regex::Error>) -> Self {
        ParseFailure { s: (*e).to_string() }.into()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("Deterministic materialization failed")]
pub struct MaterializationError();

impl<'a> NomParseError<&'a str> for Box<ParseError> {
    fn from_error_kind(_input: &'a str, kind: ErrorKind) -> Self {
        Box::new(ParseError::NomError(kind))
    }

    fn append(_input: &'a str, _kind: ErrorKind, other: Self) -> Self {
        // usually just return the existing error unchanged
        other
    }
}
