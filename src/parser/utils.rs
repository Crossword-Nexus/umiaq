use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;
use crate::umiaq_char::ALPHABET_SIZE;

pub(crate) fn letter_to_num(c: char, base_char_as_usize: usize) -> Result<usize, Box<ParseError>> {
    (c as usize).checked_sub(base_char_as_usize).and_then(|diff| {
        if diff < ALPHABET_SIZE { Some(diff) } else { None }
    }).ok_or_else(|| Box::new(ParseFailure { s : format!("Illegal char: '{c}'") }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letter_to_num_lowercase_valid() {
        // Testing with lowercase letters (a-z)
        assert_eq!(letter_to_num('a', 'a' as usize).unwrap(), 0);
        assert_eq!(letter_to_num('b', 'a' as usize).unwrap(), 1);
        assert_eq!(letter_to_num('z', 'a' as usize).unwrap(), 25);
    }

    #[test]
    fn test_letter_to_num_uppercase_valid() {
        // Testing with uppercase letters (A-Z)
        assert_eq!(letter_to_num('A', 'A' as usize).unwrap(), 0);
        assert_eq!(letter_to_num('B', 'A' as usize).unwrap(), 1);
        assert_eq!(letter_to_num('Z', 'A' as usize).unwrap(), 25);
    }

    #[test]
    fn test_letter_to_num_out_of_range() {
        // Character beyond the alphabet range
        let result = letter_to_num('[', 'A' as usize); // '[' is one after 'Z'
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Illegal char"));
    }

    #[test]
    fn test_letter_to_num_before_range() {
        // Character before the base character
        let result = letter_to_num('@', 'A' as usize); // '@' is one before 'A'
        assert!(result.is_err());
    }

    #[test]
    fn test_letter_to_num_digit() {
        // Digits should fail
        let result = letter_to_num('5', 'A' as usize);
        assert!(result.is_err());
    }

    #[test]
    fn test_letter_to_num_special_char() {
        // Special characters should fail
        let result = letter_to_num('!', 'A' as usize);
        assert!(result.is_err());
    }

    #[test]
    fn test_letter_to_num_mixed_case() {
        // Lowercase with uppercase base should fail
        let result = letter_to_num('a', 'A' as usize);
        assert!(result.is_err());

        // Uppercase with lowercase base should fail
        let result = letter_to_num('A', 'a' as usize);
        assert!(result.is_err());
    }
}
