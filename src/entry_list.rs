//! `entry_list` — Module to load and preprocess the crossword entry list for Umiaq
//!
//! This module is responsible for reading an entry list (either from a file, or from an in-memory
//! string — the latter is important for WebAssembly/browser builds, since direct file I/O
//! isn't allowed there).
//!
//! The output is an `EntryList` struct containing a flat `Vec<String>` of lowercase entries.
//! We do NOT store scores, because the solver only needs the entry strings themselves.
//!
//! The parsing logic:
//! - Each line in the input is expected to be in the format `entry;score`.
//! - Lines without a semicolon are skipped silently.
//! - `score` is parsed as an integer, and entries with scores below `min_score` are skipped.
//! - All entries are normalized to lowercase.
//! - The final list is deduplicated and sorted by length first, then alphabetically.
//!
//! This module is designed to be **WASM-friendly** — no `std::fs` calls are made unless
//! we're on a native build. The public API provides:
//! - `parse_from_str(...)` — works everywhere, including WASM.
//! - `load_from_path(...)` — **native-only** convenience method to read from a file path.
//!
//! If compiled with `target_arch = "wasm32"`, only the WASM-safe parsing method is available
//! and `parse_from_str` is not (as it's currently unused in WASM builds)

/// Struct representing a processed, ready-to-use entry list.
///
/// The `entries` vector contains all valid entries (filtered, normalized, deduplicated),
/// already sorted by (length, alphabetical).
///
/// We intentionally store just the entries themselves (`String`) because in this design
/// the solver does not require the associated scores during pattern matching.
#[derive(Debug, Clone)]
pub struct EntryList {
    /// List of lowercase entries.
    /// Example: `["able", "acid", "acorn", ...]`
    pub entries: Vec<String>,
}

impl EntryList {
    /// Parse a raw entry list from an in-memory string.
    ///
    /// This is **WASM-safe** because it doesn't touch the filesystem —
    /// you can pass the contents of a file fetched via JavaScript `fetch()` or read
    /// from the File API directly into this function.
    ///
    /// # Arguments
    /// * `contents`  — The raw file contents as a `&str`. Each line should be `entry;score`.
    /// * `min_score` — Entries with scores lower than this are skipped.
    ///
    /// # Returns
    /// * `EntryList` — Struct containing all valid entries.
    ///
    /// # Behavior:
    /// 1. Splits the input into lines.
    /// 2. Skips empty lines and lines without a `;` separator.
    /// 3. Splits each valid line into `entry` and `score` parts.
    /// 4. Parses the score and filters by `min_score`.
    /// 5. Converts `entry` to lowercase.
    /// 6. Deduplicates the list (case-insensitive because we lowercase early).
    /// 7. Sorts by length, then alphabetically.
    pub(crate) fn parse_from_str(
        contents: &str,
        min_score: i32,
    ) -> EntryList {
        // Steps 1–5: Collect valid entries into a Vec<String>.
        //
        // We use `filter_map` instead of `filter` + `map` separately
        // because it allows us to skip invalid lines in one pass.
        let mut entries: Vec<String> = contents
            .lines()
            .filter_map(|raw_line| {
                // Trim whitespace around the line.
                let line = raw_line.trim();

                // Skip empty lines early — no work needed.
                if line.is_empty() {
                    None
                // Split into two parts: `entry` and `score`.
                // Note that splitting ont he first occurrence of ';' means that entries containing
                // semicolons later (unlikely, but robust) won't break parsing.
                } else if let Some((entry_raw, score_raw)) = line.split_once(';') {
                    // Try to parse the score as an integer.
                    // If parsing fails (e.g., "abc" instead of a number), skip the line.
                    let score: i32 = score_raw.trim().parse().ok()?;

                    // Skip entries with scores below `min_score`.
                    if score < min_score {
                        None
                    } else {
                        // Convert the entry to lowercase.
                        let entry = entry_raw.trim().to_lowercase();

                        // At this point, we have a valid, normalized entry—include it.
                        Some(entry)
                    }
                } else {
                    // Skip lines without a semicolon.
                    // These are invalid because our format is `entry;score`.
                    None
                }
            })
            .collect();

        // Step 6: Deduplicate the list.
        //
        // We use sort + dedup rather than HashSet because:
        // - we need a sorted Vec anyway for the final step (sort by length)
        // - HashSet would require an additional allocation and conversion back to Vec
        // - sort + dedup is O(n log n), HashSet insert is O(n), but we save the Vec conversion
        //
        // We sort alphabetically first, because `dedup()` only removes *adjacent*
        // duplicates — and we want all duplicates next to each other.
        entries.sort();
        entries.dedup();

        // Step 7: Sort by length, then alphabetically.
        //
        // Why not do this before deduplication?
        // Because alphabetical sorting is required for `dedup()` to work properly,
        // so we have to sort twice — once alphabetically, once by (len, alpha).
        entries.sort_by(|a, b| {
            match a.len().cmp(&b.len()) {
                std::cmp::Ordering::Equal => a.cmp(b), // same length → alphabetical order
                other => other,               // otherwise sort by length
            }
        });

        // Return the final processed list wrapped in an EntryList struct.
        EntryList { entries }
    }

    /// Native-only convenience method: read from a file path and parse.
    ///
    /// This method is **not available** in WebAssembly builds, because browsers
    /// cannot read files from arbitrary paths.
    ///
    /// # Example:
    /// `let entry_list = EntryList::load_from_path("xwordlist.txt", 50, 21)?;`
    /// `println!("Loaded {} entries", entry_list.entries.len());`
    ///
    /// # Errors
    ///
    /// Will return an `Error` if unable to read a file at `path`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_path<P: AsRef<std::path::Path>>(
        path: P,
        min_score: i32,
    ) -> std::io::Result<EntryList> {
        let path_ref = path.as_ref();

        // Read the entire file into a single string.
        // Using `read_to_string` ensures UTF-8 decoding.
        let data = std::fs::read_to_string(path_ref).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("failed to read entry list from '{}': {}", path_ref.display(), e)
            )
        })?;

        // Pass the file contents to the WASM-safe parsing method.
        Ok(Self::parse_from_str(&data, min_score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let input = "cat;50\ndog;60\nbird;40";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }

    #[test]
    fn test_parse_filters_low_scores() {
        let input = "apple;100\nbanana;20\ncherry;80";
        let entry_list = EntryList::parse_from_str(input, 50);

        assert_eq!(entry_list.entries.len(), 2);
        assert_eq!(entry_list.entries, vec!["apple", "cherry"]);
    }

    #[test]
    fn test_parse_deduplicates() {
        let input = "cat;50\ndog;60\ncat;70\ncat;80";
        let entry_list = EntryList::parse_from_str(input, 45);

        // assert_eq!(entry_list.entries.len(), 2);
        // assert_eq!(entry_list.entries.iter().filter(|w| *w == "cat").count(), 1);
        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }

    #[test]
    fn test_parse_sorts_by_length_then_alpha() {
        let input = "dog;50\napple;50\ncat;50\nab;50\nzebra;50";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["ab", "cat", "dog", "apple", "zebra"]);
    }

    #[test]
    fn test_parse_normalizes_to_lowercase() {
        let input = "CAT;50\nDog;60\nBIRD;70";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["cat", "dog", "bird"]);
    }

    #[test]
    fn test_parse_skips_empty_lines() {
        let input = "cat;50\n\n\ndog;60\n\n";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }

    #[test]
    fn test_parse_skips_malformed_lines() {
        let input = "cat;50\ninvalid_line\ndog;60\nno_semicolon\napple;bad_score";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }

    #[test]
    fn test_parse_empty_input() {
        let input = "";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert!(entry_list.entries.is_empty());
    }

    #[test]
    fn test_parse_handles_whitespace() {
        let input = "  cat  ;  50  \n  dog  ;  60  ";
        let entry_list = EntryList::parse_from_str(input, 45);

        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }

    #[test]
    fn test_parse_negative_scores() {
        let input = "cat;-10\ndog;60\nbird;-5";
        let entry_list = EntryList::parse_from_str(input, 0);

        assert_eq!(entry_list.entries, vec!["dog"]);
    }

    #[test]
    fn test_parse_zero_min_score() {
        let input = "cat;0\ndog;10\nbird;-5";
        let entry_list = EntryList::parse_from_str(input, 0);

        assert_eq!(entry_list.entries, vec!["cat", "dog"]);
    }
}
