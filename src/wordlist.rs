//! wordlist.rs — Module to load and preprocess the crossword word list for Umiaq
//!
//! This module is responsible for reading a word list (either from a file, or from an in-memory
//! string — the latter is important for WebAssembly/browser builds, since direct file I/O
//! isn't allowed there).
//!
//! The output is a `WordList` struct containing a flat `Vec<String>` of lowercase words.
//! We do NOT store scores, because the solver only needs the word strings themselves.
//!
//! The parsing logic:
//! - Each line in the input is expected to be in the format `word;score`.
//! - Lines without a semicolon are skipped silently.
//! - `score` is parsed as an integer, and words with scores below `min_score` are skipped.
//! - Words longer than `max_len` are skipped.
//! - All words are normalized to lowercase.
//! - The final list is deduplicated and sorted by length first, then alphabetically.
//!
//! This module is designed to be **WASM-friendly** — no `std::fs` calls are made unless
//! we're on a native build. The public API provides:
//! - `parse_from_str(...)` — works everywhere, including WASM.
//! - `load_from_path(...)` — **native-only** convenience method to read from a file path.
//!
//! If compiled with `target_arch = "wasm32"`, only the WASM-safe parsing method is available
//! and `parse_from_str` is not (as it's currently unused in WASM builds)

/// Struct representing a processed, ready-to-use word list.
///
/// The `entries` vector contains all valid words (filtered, normalized, deduplicated),
/// already sorted by (length, alphabetical).
///
/// We intentionally store just the words themselves (`String`) because in this design
/// the solver does not require the associated scores during pattern matching.
#[derive(Debug, Clone)]
pub struct WordList {
    /// List of lowercase words.
    /// Example: `["able", "acid", "acorn", ...]`
    pub entries: Vec<String>,
}

impl WordList {
    /// Parse a raw word list from an in-memory string.
    ///
    /// This is **WASM-safe** because it doesn't touch the filesystem —
    /// you can pass the contents of a file fetched via JavaScript `fetch()` or read
    /// from the File API directly into this function.
    /// That said, we're not using this function in WASM at the moment.
    /// TODO: use this in JS instead of rewriting this logic?
    ///
    /// # Arguments
    /// * `contents`  — The raw file contents as a `&str`. Each line should be `word;score`.
    /// * `min_score` — Words with scores lower than this are skipped.
    /// * `max_len`   — Words longer than this are skipped.
    ///
    /// # Returns
    /// * `WordList` — Struct containing all valid entries.
    ///
    /// # Behavior:
    /// 1. Splits the input into lines.
    /// 2. Skips empty lines and lines without a `;` separator.
    /// 3. Splits each valid line into `word` and `score` parts.
    /// 4. Parses the score and filters by `min_score` and `max_len`.
    /// 5. Converts `word` to lowercase.
    /// 6. Deduplicates the list (case-insensitive because we lowercase early).
    /// 7. Sorts by length, then alphabetically.
    pub fn parse_from_str(
        contents: &str,
        min_score: i32,
    ) -> WordList {
        // Step 1: Collect valid words into a Vec<String>.
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
                    return None;
                }

                // Skip lines without a semicolon.
                // These are invalid because our format is `word;score`.
                if !line.contains(';') {
                    return None;
                }

                // Split into two parts: `word` and `score`.
                // `splitn(2, ';')` ensures we only split on the first semicolon,
                // so words containing semicolons later (unlikely, but robust) won't break parsing.
                let mut parts = line.splitn(2, ';');

                // Extract the word and trim extra spaces.
                let word_raw = parts.next().unwrap().trim();

                // Extract the score and trim extra spaces.
                let score_raw = parts.next().unwrap().trim();

                // Try to parse the score as an integer.
                // If parsing fails (e.g., "abc" instead of a number), skip the line.
                let score: i32 = score_raw.parse().ok()?;

                // Skip words with scores below `min_score`.
                if score < min_score {
                    return None;
                }

                // Convert the word to lowercase.
                let word = word_raw.to_lowercase();

                // At this point, we have a valid, normalized word — include it.
                Some(word)
            })
            .collect();

        // Step 2: Deduplicate the list.
        //
        // We sort alphabetically first, because `dedup()` only removes *adjacent*
        // duplicates — and we want all duplicates next to each other.
        entries.sort();
        entries.dedup();

        // Step 3: Sort by length, then alphabetically.
        //
        // Why not do this before deduplication?
        // Because alphabetical sorting is required for `dedup()` to work properly,
        // so we have to sort twice — once alphabetically, once by (len, alpha).
        entries.sort_by(|a, b| {
            match a.len().cmp(&b.len()) {
                std::cmp::Ordering::Equal => a.cmp(b), // same length → alphabetical order
                other => other,     // otherwise sort by length
            }
        });

        // Return the final processed list wrapped in a WordList struct.
        WordList { entries }
    }

    /// Native-only convenience method: read from a file path and parse.
    ///
    /// This method is **not available** in WebAssembly builds, because browsers
    /// cannot read files from arbitrary paths.
    ///
    /// # Example:
    /// `let wl = WordList::load_from_path("xwordlist.txt", 50, 21)?;`
    /// `println!("Loaded {} words", wl.entries.len());`
    ///
    /// # Errors
    ///
    /// Will return an `Error` if unable to read a file at `path`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from_path<P: AsRef<std::path::Path>>(
        path: P,
        min_score: i32,
    ) -> std::io::Result<WordList> {
        // Read the entire file into a single string.
        // Using `read_to_string` ensures UTF-8 decoding.
        let data = std::fs::read_to_string(path)?;

        // Pass the file contents to the WASM-safe parsing method.
        Ok(Self::parse_from_str(&data, min_score))
    }
}
