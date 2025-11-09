use crate::comparison_operator::ComparisonOperator::{EQ, GE, GT, LE, LT, NE};
use crate::errors::ParseError::ParseFailure;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::{LazyLock, OnceLock};

// pub(crate) static COMPARISON_OPERATORS: [ComparisonOperator; 6] = [EQ,NE,LE,GE,LT,GT];

#[derive(Clone, Copy)]
pub(crate) enum ComparisonOperator {
    EQ,
    NE,
    LE,
    GE,
    LT,
    GT
}

impl ComparisonOperator {
    /// All operator variants, used to build the reverse lookup map.
    pub(crate) const ALL: [ComparisonOperator; 6] = [EQ, NE, LE, GE, LT, GT];

    /// All operator strings, lazily initialized.
    pub(crate) fn all_as_strings() -> &'static Vec<String> {
        static ALL_AS_STRINGS: LazyLock<Vec<String>> = LazyLock::new(|| {
            ComparisonOperator::ALL
                .iter()
                .map(ComparisonOperator::to_string)
                .collect()
        });
        &ALL_AS_STRINGS
    }

    /// Returns the string representation of this operator.
    /// Single source of truth for operator-to-string mapping.
    const fn as_str(self) -> &'static str {
        match self {
            EQ => "=",
            NE => "!=",
            LE => "<=",
            GE => ">=",
            LT => "<",
            GT => ">"
        }
    }
}

/// Lazily initialized `HashMap` for constant-time string-to-operator lookup.
/// Built from `as_str()` to ensure a single source of truth.
///
/// NB: This approach (`LazyLock` + `HashMap`) is necessary because const `HashMap`
/// construction is not yet stable in Rust. Once `const_trait_impl` is stabilized,
/// this could potentially be simplified.
static OP_MAP: OnceLock<HashMap<&'static str, ComparisonOperator>> = OnceLock::new();

fn get_str_to_op_map() -> &'static HashMap<&'static str, ComparisonOperator> {
    OP_MAP.get_or_init(|| {
        ComparisonOperator::ALL
            .iter()
            .map(|op| (op.as_str(), *op))
            .collect()
    })
}

impl FromStr for ComparisonOperator {
    type Err = crate::errors::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        get_str_to_op_map()
            .get(s)
            .copied()
            .ok_or_else(|| {
                let expected = ComparisonOperator::all_as_strings().join(", ");
                ParseFailure {
                    s: format!("Invalid comparison operator '{s}' (expected: {expected})")
                }
            })
    }
}

impl fmt::Display for ComparisonOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
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

    #[test]
    fn test_round_trip_all_operators() {
        let operators = vec!["=", "!=", "<=", ">=", "<", ">"];

        for op_str in operators {
            let parsed = op_str.parse::<ComparisonOperator>().unwrap();
            let displayed = parsed.to_string();
            assert_eq!(op_str, displayed, "Round-trip failed for '{}'", op_str);
        }
    }

    #[test]
    fn test_from_str_invalid_operators() {
        let invalid = vec!["==", "equals", "", " = ", "!", "<>", "=>"];

        for input in invalid {
            let result = input.parse::<ComparisonOperator>();
            assert!(result.is_err(), "Should reject invalid operator '{}'", input);
        }
    }
}
