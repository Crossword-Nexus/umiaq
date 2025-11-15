//! Integration tests for the Umiaq crossword solver.
//!
//! These tests verify the complete pipeline from equation parsing through solving
//! to result validation, using realistic entry lists and complex constraint scenarios.

use std::collections::HashSet;
use std::fs;

use umiaq::solver::{solve_equation, SolveResult, SolveStatus, SolverError};
use umiaq::errors::ParseError;

/// Load the test entry
/// list from fixtures
fn load_test_entry_list() -> Vec<String> {
    let content = fs::read_to_string("tests/fixtures/test_entry_list.txt")
        .expect("Failed to read test entry list");

    content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.split(';')
                .next()
                .expect("Invalid entry-list format")
                .to_lowercase()
        })
        .collect()
}

/// Helper to convert Vec<String> to Vec<&str>
fn as_str_slice(entries: &[String]) -> Vec<&str> {
    entries.iter().map(|s| s.as_str()).collect()
}

/// Helper to extract just the entries from a solution
fn solution_entries(result: &SolveResult) -> Vec<Vec<String>> {
    result
        .solutions
        .iter()
        .map(|bindings_vec| {
            bindings_vec
                .iter()
                .filter_map(|b| b.get_entry().map(|rc| rc.to_string()))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod simple_equations {
    use super::*;

    #[test]
    fn test_single_literal_pattern() {
        let entries = vec!["able", "acid", "area"];
        let result = solve_equation("able", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_entries(&result), vec![vec!["able"]]);
        // Entry list is exhausted after finding the one match
        assert_eq!(result.status, SolveStatus::EntryListExhausted);
    }

    #[test]
    fn test_single_variable_pattern() {
        let entries = vec!["a", "be", "cat", "dogs"];
        let result = solve_equation("A", &entries, 10).unwrap();

        // Should match all 4 entries
        assert_eq!(result.solutions.len(), 4);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert!(entries_found.contains("a"));
        assert!(entries_found.contains("be"));
        assert!(entries_found.contains("cat"));
        assert!(entries_found.contains("dogs"));
    }

    #[test]
    fn test_pattern_with_wildcard() {
        let entries = vec!["cat", "bat", "rat", "car", "bar"];
        let result = solve_equation(".at", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 3);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from(["cat".to_string(), "bat".to_string(), "rat".to_string()]));
    }

    #[test]
    fn test_pattern_with_literal_and_variable() {
        let entries = vec!["able", "axle", "agile", "ankle", "angle"];
        let result = solve_equation("aAle", &entries, 10).unwrap();

        // All 5 entries match: a[X]le where X can be b, x, gi, nk, or ng
        assert_eq!(result.solutions.len(), 5);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from([
            "able".to_string(),
            "axle".to_string(),
            "agile".to_string(),
            "ankle".to_string(),
            "angle".to_string()
        ]));
    }

    #[test]
    fn test_even_length_palindrome_pattern() {
        let entries = vec!["noon", "deed", "level", "radar", "test"];
        let result = solve_equation("A~A", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 2);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from(["noon".to_string(), "deed".to_string()]));
    }

    #[test]
    fn test_anagram_pattern() {
        let entry_list = load_test_entry_list();

        // Verify "area" exists in entry list (it's in our test fixtures)
        assert!(entry_list.contains(&"area".to_string()), "Test entry list should contain 'area'");

        let entries = as_str_slice(&entry_list);

        // /area is anagram pattern - should find entries with letters a,e,r,a
        let result = solve_equation("/area", &entries, 10).unwrap();

        // At minimum, "area" itself should be found
        assert!(result.solutions.len() > 0);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert!(entries_found.contains("area"));
    }
}

#[cfg(test)]
mod multi_pattern_equations {
    use super::*;

    #[test]
    fn test_two_patterns_shared_variable() {
        let entries = vec!["ab", "a", "ba", "b"];
        let result = solve_equation("Ab;A", &entries, 10).unwrap();

        // "Ab" means A followed by literal 'b'
        // Only "ab" matches with A="a"
        // Then "A" with A="a" matches entry "a"
        // Solution: ["ab", "a"]
        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_entries(&result), vec![vec!["ab", "a"]]);
    }

    #[test]
    fn test_three_patterns_chain() {
        let entries = vec!["ab", "ba", "bc", "cb"];
        let result = solve_equation("AB;BA;BC", &entries, 10).unwrap();

        // Should find exactly 6 solutions:
        // ab•ba•ba, ab•ba•bc, ba•ab•ab, bc•cb•cb, cb•bc•ba, cb•bc•bc
        assert_eq!(result.solutions.len(), 6);

        let solutions = solution_entries(&result);
        let solutions_set: HashSet<Vec<String>> = solutions.into_iter().collect();

        assert!(solutions_set.contains(&vec!["ab".to_string(), "ba".to_string(), "ba".to_string()]));
        assert!(solutions_set.contains(&vec!["ab".to_string(), "ba".to_string(), "bc".to_string()]));
        assert!(solutions_set.contains(&vec!["ba".to_string(), "ab".to_string(), "ab".to_string()]));
        assert!(solutions_set.contains(&vec!["bc".to_string(), "cb".to_string(), "cb".to_string()]));
        assert!(solutions_set.contains(&vec!["cb".to_string(), "bc".to_string(), "ba".to_string()]));
        assert!(solutions_set.contains(&vec!["cb".to_string(), "bc".to_string(), "bc".to_string()]));
    }

    #[test]
    fn test_multiple_solutions() {
        let entries = vec!["inch", "chin"];
        let result = solve_equation("AB;BA", &entries, 10).unwrap();

        // inch/chin: A=in, B=ch OR A=ch, B=in
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_entries(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;

    #[test]
    fn test_length_constraint_exact() {
        let entries = vec!["a", "ab", "abc", "abcd"];
        let result = solve_equation("A;|A|=2", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_entries(&result), vec![vec!["ab"]]);
    }

    #[test]
    fn test_length_constraint_range() {
        let entries = vec!["a", "ab", "abc", "abcd", "abcde"];
        let result = solve_equation("A;|A|>=2;|A|<=3", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 2);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from(["ab".to_string(), "abc".to_string()]));
    }

    #[test]
    fn test_inequality_constraint() {
        let entries = vec!["inch", "chin", "noon", "deed"];
        let result = solve_equation("AB;BA;!=AB", &entries, 10).unwrap();

        // Should find inch/chin and chin/inch but not noon/noon or deed/deed
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_entries(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }

    #[test]
    fn test_form_constraint() {
        let entries = vec!["able", "axle", "area", "beta"];
        let result = solve_equation("A;A=(a*)", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 3);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from(["able".to_string(), "axle".to_string(), "area".to_string()]));
    }

    #[test]
    fn test_complex_constraint() {
        let entries = vec!["a", "ab", "abc", "ace", "are", "be"];
        let result = solve_equation("A;A=(2-3:a*)", &entries, 10).unwrap();

        // Length 2-3, starts with 'a'
        // Should match: ab, abc, ace, are
        // Should NOT match: a (too short), be (doesn't start with 'a')
        assert_eq!(result.solutions.len(), 4);
        let entries_found: HashSet<String> = solution_entries(&result)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(entries_found, HashSet::from(["ab".to_string(), "abc".to_string(), "ace".to_string(), "are".to_string()]));
    }

    #[test]
    fn test_joint_constraint() {
        let entries = vec!["inch", "chin", "chess", "test", "able"];
        let result = solve_equation("AB;BA;|AB|=4", &entries, 10).unwrap();

        // inch (i+nch=4) + chin (ch+in=4) ✓
        // chin (ch+in=4) + inch (in+ch=4) ✓
        assert_eq!(result.solutions.len(), 2);
        let solutions_set: HashSet<Vec<String>> = solution_entries(&result).into_iter().collect();
        assert!(solutions_set.contains(&vec!["inch".to_string(), "chin".to_string()]));
        assert!(solutions_set.contains(&vec!["chin".to_string(), "inch".to_string()]));
    }

    #[test]
    fn test_multiple_constraints_combined() {
        let entry_list = load_test_entry_list();
        let entries = as_str_slice(&entry_list);

        // Pattern with multiple constraint types: length 4, starts with 'b'
        // Note: Can't have multiple conflicting form constraints on the same variable
        let result = solve_equation("A;|A|=4;A=(b*)", &entries, 10).unwrap();

        // Verify all results match constraints
        let solutions = solution_entries(&result);
        assert!(!solutions.is_empty(), "Should find at least one solution");

        for sol in &solutions {
            for entry in sol {
                assert_eq!(entry.len(), 4, "Entry '{}' should be length 4", entry);
                assert!(entry.starts_with('b'), "Entry '{}' should start with 'b'", entry);
            }
        }
    }
}

#[cfg(test)]
mod error_cases {
    use super::*;

    #[test]
    fn test_parse_error_invalid_syntax() {
        let entries = vec!["test"];
        let result = solve_equation("A;|A|", &entries, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_contradictory_bounds() {
        let entries = vec!["test"];
        let result = solve_equation("A;|A|>10;|A|<5", &entries, 10);

        // Should fail during parsing due to contradictory constraints
        assert!(result.is_err());
        let e = result.unwrap_err();
        // Verify it's a parse error containing contradictory bounds
        if let SolverError::ParseFailure(parse_err) = e {
            assert!(matches!(*parse_err, ParseError::ContradictoryBounds { min: 11, max: 4 }));
        } else {
            panic!("Expected ParseFailure error, got: {:?}", e);
        }
    }

    #[test]
    fn test_no_solutions_found() {
        let entries = vec!["able", "baker"];
        let result = solve_equation("xyz", &entries, 10).unwrap();

        // No matches for literal "xyz"
        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::EntryListExhausted);
    }

    #[test]
    fn test_empty_entry_list() {
        let entries: Vec<&str> = vec![];
        let result = solve_equation("A", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::EntryListExhausted);
    }

    #[test]
    fn test_no_matching_entries_for_constraints() {
        let entries = vec!["test", "able", "entry"];
        // Looking for an entry ending with 'x' and an entry starting with 'x'
        // None exist in this entry list
        let result = solve_equation("Ax;xB", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 0);
        assert_eq!(result.status, SolveStatus::EntryListExhausted);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_very_long_pattern() {
        let entries = vec!["abcdefghij"];
        let result = solve_equation("ABCDEFGHIJ", &entries, 10).unwrap();

        assert_eq!(result.solutions.len(), 1);
        assert_eq!(solution_entries(&result), vec![vec!["abcdefghij"]]);
    }

    #[test]
    fn test_many_patterns() {
        let entries = vec!["a", "b", "c"];
        // 10 patterns, each matching single letter
        let result = solve_equation("A;B;C;D;E;F;G;H;I;J", &entries, 10).unwrap();

        // 3^10 = 59049 total combinations, but we only request 10
        assert_eq!(result.solutions.len(), 10);
        assert_eq!(result.status, SolveStatus::FoundEnough);
    }

    #[test]
    fn test_early_stop_on_limit() {
        let entries = vec!["a", "b", "c", "d", "e"];
        let result = solve_equation("A", &entries, 3).unwrap();

        // Should stop at exactly 3 results
        assert_eq!(result.solutions.len(), 3);
        assert_eq!(result.status, SolveStatus::FoundEnough);
    }

    #[test]
    fn test_entry_list_exhausted() {
        let entries = vec!["a", "b"];
        let result = solve_equation("A", &entries, 100).unwrap();

        // Requested 100 but only 2 available
        assert_eq!(result.solutions.len(), 2);
        assert_eq!(result.status, SolveStatus::EntryListExhausted);
    }
}

#[cfg(test)]
mod realistic_scenarios {
    use super::*;

    #[test]
    fn test_crossword_intersection() {
        let entry_list = load_test_entry_list();
        let entries = as_str_slice(&entry_list);

        // Simulates two crossing entries in a crossword
        // Pattern 1: bABC (literal 'b' + variables A, B, C)
        // Pattern 2: ADEF (variables A, D, E, F)
        // Both patterns share variable A
        let result = solve_equation("bABC;ADEF", &entries, 5).unwrap();

        // Verify all solutions have the correct intersection
        let solutions = solution_entries(&result);
        assert!(!solutions.is_empty(), "Should find at least one solution");

        for sol in &solutions {
            assert_eq!(sol.len(), 2, "Each solution should have 2 entries");
            let first = &sol[0];
            let second = &sol[1];
            assert!(first.starts_with('b'), "First entry should start with 'b'");
            // First letter of second entry should match second letter of first entry (both are A)
            assert_eq!(
                second.chars().nth(0),
                first.chars().nth(1),
                "Second entry '{}' first letter should match first entry '{}' second letter",
                second,
                first
            );
        }
    }

    #[test]
    fn test_themed_puzzle() {
        let entry_list = load_test_entry_list();
        let entries = as_str_slice(&entry_list);

        // All entries must start with 'b' (themed)
        let result = solve_equation("bA;bB;bC", &entries, 3).unwrap();

        let solutions = solution_entries(&result);
        assert_eq!(solutions.len(), 3);

        for sol in solutions {
            assert_eq!(sol.len(), 3, "Each solution should have 3 entries");
            // Verify all entries in each solution start with 'b'
            assert!(sol.iter().all(|w| w.starts_with('b')));
        }
    }

    #[test]
    fn test_complex_real_world() {
        let entry_list = load_test_entry_list();
        let entries = as_str_slice(&entry_list);

        // Complex equation with multiple patterns and constraints
        // AB and BA with |A|>=2, |B|>=2, !=AB
        // This finds pairs like "inch"/"chin" where the parts can be swapped
        let result = solve_equation(
            "AB;BA;|A|>=2;|B|>=2;!=AB",
            &entries,
            10
        ).unwrap();

        // Should find at least some solutions
        assert!(!result.solutions.is_empty(), "Should find at least one solution");

        // Verify each solution has 2 bindings (one for each pattern)
        for solution in &result.solutions {
            assert_eq!(solution.len(), 2, "Each solution should have 2 bindings");
        }

        // Verify we got actual entries in the solutions
        let solutions = solution_entries(&result);
        for sol in &solutions {
            assert_eq!(sol.len(), 2, "Each solution should have 2 entries");
            // Both entries should exist in our entry list
            assert!(entry_list.iter().any(|w| w == &sol[0]), "Entry '{}' should be in entry list", sol[0]);
            assert!(entry_list.iter().any(|w| w == &sol[1]), "Entry '{}' should be in entry list", sol[1]);
        }
    }
}
