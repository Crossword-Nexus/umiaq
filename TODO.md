# Umiaq TODO & Future Ideas

This list tracks potential features and optimizations inspired by comparative analysis with other wordplay tools (like `word-finder.html`).

## Search Primitives
- [ ] **Neighbor Operator (`` ` ``)**: Hamming/Levenshtein distance 1 (e.g., `` `bonge ``).
    - **Complexity: Moderate**. Requires a new `FormPart` variant, parser updates, and a custom matching loop in `matcher.rs`. Regex prefiltering would be broad (just length-based).
- [ ] **Fuzzy Anagrams**: Support wildcards/extras in anagrams (e.g., `/abc..`, `/abc*`).
    - **Complexity: Moderate**. Requires updating the `Alphagram` logic to handle "at least" counts and reserved character slots for wildcards.
- [ ] **Length-Bounded Patterns**: Shorthand prefixes (e.g., `8:A*B`).
    - **Complexity: Low-Moderate**. Mostly a parser-level convenience that desugars into existing `JointConstraints`.

## Syntax Ergonomics
- [ ] **Inline Variable Definitions**: Define structure in-place (e.g., `A=(#@#)`).
    - **Complexity: Low**. Simple syntax sugar to map `(...)` directly to the `VarConstraint.form` property.
- [ ] **Set-based Constraints**: Shorthand for "one of these words" (e.g., `A={cat,dog}`).
    - **Complexity: Moderate**. Requires a new constraint type and efficient `HashSet` lookups during the materialization/joining phase.

## Optimization
- [ ] **Dynamic Pattern Ordering**: Use dictionary distribution for greedy ordering.
    - **Complexity: High**. Requires `get_ordered_patterns` to access `ScanHints` or the `EntryList`. This is a significant architectural change but could prevent exponential blowup in complex queries.
- [ ] **Bitset-based Prefiltering**: Use bitmasks for vowels/consonants during `scan_batch`.
    - **Complexity: Moderate-High**. Touching the inner loop of the scanner. Requires pre-computing bitmasks for the entire dictionary to allow O(1) rejection of vowel/consonant patterns.

## UI/WASM
- [ ] **Breakdown Display**: Show variable splits in the results (e.g., `C·A·T`).
    - **Complexity: Low-Moderate**. The `Bindings` already contain the data; the WASM interface just needs to expose the segments or provide a "re-match" utility for the frontend.
