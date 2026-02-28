use std::collections::HashSet;
use std::ops::RangeInclusive;
use std::sync::LazyLock;

// Character-set constants
pub(crate) const ALPHABET_SIZE: usize = 26;
pub(crate) const LOWERCASE_ALPHABET: RangeInclusive<char> = 'a'..='z';
#[cfg(test)]
pub(crate) const UPPERCASE_ALPHABET: RangeInclusive<char> = 'A'..='Z';

pub(crate) const VOWELS: &str = "aeiouy";
pub(crate) const CONSONANTS: &str = "bcdfghjklmnpqrstvwxz";
pub(crate) const VARIABLE_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub(crate) const LITERAL_CHARS: &str = "abcdefghijklmnopqrstuvwxyz";

pub(crate) const NUM_POSSIBLE_VARIABLES: usize = VARIABLE_CHARS.len();

static VOWEL_SET: LazyLock<HashSet<char>> = LazyLock::new(|| VOWELS.chars().collect());
static CONSONANT_SET: LazyLock<HashSet<char>> = LazyLock::new(|| CONSONANTS.chars().collect());

pub(crate) trait UmiaqChar {
    fn is_vowel(&self) -> bool;
    fn is_consonant(&self) -> bool;
    fn is_variable(&self) -> bool;
    fn is_literal(&self) -> bool;
}

impl UmiaqChar for char {
    fn is_vowel(&self) -> bool {
        VOWEL_SET.contains(self)
    }
    fn is_consonant(&self) -> bool {
        CONSONANT_SET.contains(self)
    }
    fn is_variable(&self) -> bool {
        self.is_ascii_uppercase()
    }
    fn is_literal(&self) -> bool {
        self.is_ascii_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_vowel() {
        assert!('a'.is_vowel());
        assert!('e'.is_vowel());
        assert!('i'.is_vowel());
        assert!('o'.is_vowel());
        assert!('u'.is_vowel());
        assert!('y'.is_vowel());
    }

    #[test]
    fn test_is_not_vowel() {
        assert!(!'b'.is_vowel());
        assert!(!'z'.is_vowel());
        assert!(!'A'.is_vowel()); // uppercase
        assert!(!'1'.is_vowel());
        assert!(!'@'.is_vowel());
    }

    #[test]
    fn test_is_consonant() {
        assert!('b'.is_consonant());
        assert!('c'.is_consonant());
        assert!('z'.is_consonant());
    }

    #[test]
    fn test_is_not_consonant() {
        assert!(!'a'.is_consonant());
        assert!(!'e'.is_consonant());
        assert!(!'B'.is_consonant()); // uppercase
        assert!(!'1'.is_consonant());
        assert!(!'.'.is_consonant());
    }

    #[test]
    fn test_vowel_consonant_mutual_exclusivity() {
        for c in LOWERCASE_ALPHABET {
            let is_v = c.is_vowel();
            let is_c = c.is_consonant();
            assert_ne!(is_v, is_c, "char '{}' should be either vowel or consonant, not both or neither", c);
        }
    }

    #[test]
    fn test_is_variable() {
        assert!('A'.is_variable());
        assert!('Z'.is_variable());
        assert!('M'.is_variable());
    }

    #[test]
    fn test_is_not_variable() {
        assert!(!'a'.is_variable()); // lowercase
        assert!(!'1'.is_variable());
        assert!(!'@'.is_variable());
    }

    #[test]
    fn test_is_literal() {
        assert!('a'.is_literal());
        assert!('z'.is_literal());
        assert!('m'.is_literal());
    }

    #[test]
    fn test_is_not_literal() {
        assert!(!'A'.is_literal()); // uppercase
        assert!(!'1'.is_literal());
        assert!(!'@'.is_literal());
    }

    #[test]
    fn test_variable_literal_mutual_exclusivity() {
        for c in UPPERCASE_ALPHABET {
            assert!(c.is_variable());
            assert!(!c.is_literal());
        }
        for c in LOWERCASE_ALPHABET {
            assert!(!c.is_variable());
            assert!(c.is_literal());
        }
    }

    #[test]
    fn test_alphabet_constants() {
        assert_eq!(ALPHABET_SIZE, 26);
        assert_eq!(VOWELS.len(), 6); // a e i o u y
        assert_eq!(CONSONANTS.len(), 20); // 26 - 6
        assert_eq!(VARIABLE_CHARS.len(), 26);
        assert_eq!(LITERAL_CHARS.len(), 26);
    }
}
