# Umiaq

**Umiaq** is an open-source solver and word-pattern matching tool.
Umiaq is designed for crossword constructors, wordplay enthusiasts, and puzzle makers who want to search large word lists using expressive patterns, variables, and constraints.

👉 [Try the web interface](https://crossword-nexus.github.io/umiaq/)

📝 [Technical documentation](https://crossword-nexus.github.io/umiaq/docs/umiaq/)

---

## Features

- **Expressive pattern matching**  
  Match words against patterns with variables, wildcards, and constraints.

- **Variable binding**  
  Use uppercase letters (`A`–`Z`) as variables that can bind to substrings and be reused.

- **Wildcards**
  - `.`: any single letter
  - `*`: zero or more letters
  - `@`: any vowel (`aeiouy`)
  - `#`: any consonant (`bcdfghjklmnpqrstvwxz`)
  - `[abc]`: any of the listed letters (here: `a`, `b`, or `c`)
  - `[a-e]`: any letter in the range from `a` to `e` (`a`, `b`, `c`, `d`, `e`)
    - Ranges and lists can be combined: `[abcw-z]`
  - `/abc`: any anagram of the listed letters (here: `a`, `b`, or `c`)

- **Constraints**  
  Add conditions on variables or groups of variables:
  - length: `|A|=3`
  - inequality: `!=ABC` (`A`, `B`, and `C` must all be distinct)
  - complex: `A=(3-5:a*)` (length 3–5, must match pattern `a*`)
  - joint: `|ABC|=10` (the lengths of `A`, `B`, and `C` sum to 10)

- **Reversed variables**  
  `~A` matches the reverse of variable `A`.

---

## Examples

- `l.x` → words like **lax**, **lox**
- `gr[a-e]y` → matches **gray**, **grey**
- `A~A` → even-length palindromes like **noon**, **redder**
- `A.~A` → odd-length palindromes like **non**, **radar**
- `AB;|A|=2;|B|=2;!=AB` → 4-letters words made up of two concatenated distinct 2-letter substrings
- `A@#A` → words with some string, then a vowel, then a consonant, then the initial string again
- `/triangle` → any anagram of "triangle"
- `A;AB;|AB|=7;A=(3-4:g*)` → 7-letter words starting with a 3–to-4-letter word that begins with **g**

## Development

- Build the web version: `wasm-pack build --release --target web --out-dir web/pkg`
- Run tests: `cargo test`
- Run benchmarks: `cargo run --bin bench_local --release -- -r 1 -p 0`

---

## License

- **Code**: MIT License
- **Word list (Spread the Word List)**: CC BY-NC-SA 4.0  

