use crate::comparison_operator::ComparisonOperator::{EQ, GE, GT, LE, LT, NE};
use crate::errors::ParseError::ParseFailure;
use std::fmt;
use std::str::FromStr;

// pub(crate) static COMPARISON_OPERATORS: [ComparisonOperator; 6] = [EQ,NE,LE,GE,LT,GT];

#[derive(Clone)]
pub(crate) enum ComparisonOperator {
    EQ,
    NE,
    LE,
    GE,
    LT,
    GT
}

impl FromStr for ComparisonOperator {
    type Err = crate::errors::ParseError;

    // TODO? DRY w/Display::fmt
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "=" => Ok(EQ),
            "!=" => Ok(NE),
            "<=" => Ok(LE),
            ">=" => Ok(GE),
            "<" => Ok(LT),
            ">" => Ok(GT),
            _ => Err(ParseFailure {
                s: format!("Invalid comparison operator '{}' (expected: =, !=, <, >, <=, >=)", s)
            })
        }
    }
}

impl fmt::Display for ComparisonOperator {
    // TODO? DRY w/FromStr::from_str
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            EQ => "=",
            NE => "!=",
            LE => "<=",
            GE => ">=",
            LT => "<",
            GT => ">"
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_string() {
        assert_eq!(EQ.to_string(), "=");
        assert_eq!(NE.to_string(), "!=");
        assert_eq!(LE.to_string(), "<=");
        assert_eq!(GE.to_string(), ">=");
        assert_eq!(LT.to_string(), "<");
        assert_eq!(GT.to_string(), ">");
    }
}
