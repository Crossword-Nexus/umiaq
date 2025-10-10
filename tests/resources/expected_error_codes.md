# Error Code Reference

**⚠️ This document is auto-generated from the source code. Do not edit manually.**

## Table of Contents

- [Solver Errors (S001–S003)](#solver-errors)
- [Parse Errors (E001–E016)](#parse-errors)
- [How to Use Error Codes](#how-to-use-error-codes)

## Solver Errors

Top-level errors from the solver. These wrap lower-level parse errors or indicate solver-specific issues.

### S001: Pattern parsing failed

**Details:** The input pattern could not be parsed. This wraps an underlying ParseError (see Parse Errors section for specific error codes).

**Example error message:**
```
parse failure: Empty form string
```

**Detailed format:**
```
S001
  caused by: Empty form string (E003)
Example: Use 'A*B' or '*cat*' instead of empty string
```

---

### S002: Equation has only constraints, no patterns to solve

**Details:** The equation contains only constraints (like `|A|=3`) but no actual patterns to match against words. Add at least one pattern like `A*B` or `*cat*`.

**How to fix:**
```
Add at least one pattern to solve. Example: 'A*B' or '*cat*;*dog*'
```

**Example error message:**
```
no patterns to solve (constraints only)
```

**Detailed format:**
```
no patterns to solve (constraints only) (S002)
Add at least one pattern to solve. Example: 'A*B' or '*cat*;*dog*'
```

---

### S003: Internal error during solution construction

**Details:** This indicates an internal solver error where a pattern matched but constraints could not be satisfied during solution materialization. This is usually a bug in the solver logic.

**How to fix:**
```
This is an internal error. The pattern matched but constraints could not be satisfied.
```

**Example error message:**
```
materialization error: pattern depth: 2/3, environment: {A: "test"}
```

**Detailed format:**
```
materialization error: pattern depth: 2/3, environment: {A: "test"} (S003)
This is an internal error. The pattern matched but constraints could not be satisfied.
```

---

## Parse Errors

Errors that occur when parsing pattern strings, constraints, or equations.

### E001: Generic parse failure

**Details:** The form string could not be parsed as any recognized pattern type.

**Example error message:**
```
Form parsing failed: "BAD(INPUT"
```

**Detailed format:**
```
Form parsing failed: "BAD(INPUT" (E001)
```

---

### E002: Invalid regex pattern

**Details:** A regular expression pattern used internally failed to compile or execute.

**Example error message:**
```
Invalid regex pattern: Parsing error at position 1: Could not parse group name
```

**Detailed format:**
```
Invalid regex pattern: Parsing error at position 1: Could not parse group name (E002)
```

---

### E003: Empty form string

**Details:** The pattern string is empty. At least one pattern is required.

**How to fix:**
```
Example: Use 'A*B' or '*cat*' instead of empty string
```

**Example error message:**
```
Empty form string
```

**Detailed format:**
```
Empty form string (E003)
Example: Use 'A*B' or '*cat*' instead of empty string
```

---

### E004: Invalid length range format

**Details:** Length ranges must be in the format N-M where N ≤ M.

**How to fix:**
```
Expected format: N-M where N ≤ M (e.g., '3-5' or '1-10')
```

**Example error message:**
```
Invalid length range: "10-5"
```

**Detailed format:**
```
Invalid length range: "10-5" (E004)
Expected format: N-M where N ≤ M (e.g., '3-5' or '1-10')
```

---

### E005: Invalid complex constraint

**Details:** The complex constraint syntax is invalid or malformed.

**Example error message:**
```
bad constraint
```

**Detailed format:**
```
bad constraint (E005)
```

---

### E006: Invalid input

**Details:** The input does not match any expected format for patterns or constraints.

**Example error message:**
```
Invalid input: bad input
```

**Detailed format:**
```
Invalid input: bad input (E006)
```

---

### E007: Integer parsing error

**Details:** A numeric value in the pattern could not be parsed as an integer.

**Example error message:**
```
int-parsing error: invalid digit found in string
```

**Detailed format:**
```
int-parsing error: invalid digit found in string (E007)
```

---

### E008: Contradictory length bounds

**Details:** The minimum length constraint exceeds the maximum length constraint, making it impossible to satisfy.

**How to fix:**
```
The minimum length cannot exceed the maximum length
```

**Example error message:**
```
contradictory bounds: min=5, max=3
```

**Detailed format:**
```
contradictory bounds: min=5, max=3 (E008)
The minimum length cannot exceed the maximum length
```

---

### E009: Invalid charset range

**Details:** In a charset range like `[a-z]`, the first character must come before the second in ASCII order.

**How to fix:**
```
In a charset range, the first character must come before the second (e.g., 'a-z' not 'z-a')
```

**Example error message:**
```
Invalid range in charset: z-a
```

**Detailed format:**
```
Invalid range in charset: z-a (E009)
In a charset range, the first character must come before the second (e.g., 'a-z' not 'z-a')
```

---

### E010: Dangling '-' in charset

**Details:** A charset ends with a dash but no closing character for the range.

**How to fix:**
```
Remove the trailing '-' or complete the range (e.g., '[abc]' or '[a-c]')
```

**Example error message:**
```
Dangling '-' at end of charset
```

**Detailed format:**
```
Dangling '-' at end of charset (E010)
Remove the trailing '-' or complete the range (e.g., '[abc]' or '[a-c]')
```

---

### E011: Conflicting variable constraints

**Details:** A variable has multiple incompatible constraints that cannot both be satisfied.

**How to fix:**
```
Variable has incompatible constraints that cannot both be satisfied
```

**Example error message:**
```
Conflicting constraints for A (|A|=3 / |A|=5)
```

**Detailed format:**
```
Conflicting constraints for A (|A|=3 / |A|=5) (E011)
Variable has incompatible constraints that cannot both be satisfied
```

---

### E012: Parse error in clause

**Details:** One of the clauses in a multi-clause equation could not be parsed. This wraps an underlying ParseError.

**Example error message:**
```
Parse error in clause 'BAD(INPUT': Invalid input: BAD(INPUT
```

**Detailed format:**
```
Parse error in clause 'BAD(INPUT': Invalid input: BAD(INPUT (E012)
```

---

### E013: Variable name not A-Z

**Details:** Variable names must be single uppercase letters from A to Z.

**How to fix:**
```
Variable names must be single uppercase letters A-Z
```

**Example error message:**
```
Invalid variable name '1' (must be A-Z)
```

**Detailed format:**
```
Invalid variable name '1' (must be A-Z) (E013)
Variable names must be single uppercase letters A-Z
```

---

### E014: Non-lowercase character in pattern

**Details:** Only lowercase letters a-z are allowed in this context (typically in literal strings or anagrams).

**How to fix:**
```
Only lowercase letters a-z are allowed
```

**Example error message:**
```
Invalid character 'X' (only lowercase a-z allowed)
```

**Detailed format:**
```
Invalid character 'X' (only lowercase a-z allowed) (E014)
Only lowercase letters a-z are allowed
```

---

### E015: Invalid characters in anagram

**Details:** Anagram constraints must contain only lowercase letters a-z.

**How to fix:**
```
Anagrams must contain only lowercase letters a-z
```

**Example error message:**
```
Anagram constraint "XYZ" contains invalid character 'X' (only a-z allowed)
```

**Detailed format:**
```
Anagram constraint "XYZ" contains invalid character 'X' (only a-z allowed) (E015)
Anagrams must contain only lowercase letters a-z
```

---

### E016: Low-level parser error

**Details:** An error occurred in the low-level nom parser. This typically indicates malformed input at the character level.

**Example error message:**
```
nom parser error: Tag
```

**Detailed format:**
```
nom parser error: Tag (E016)
```

---


## How to Use Error Codes

When you see an error like:

```
Error: Empty form string (E003)
Example: Use 'A*B' or '*cat*' instead of empty string
```

1. Note the error code (e.g., `E003`)
2. Look it up in this document for detailed explanation
3. Follow the suggested resolution steps

## Error Display Formats

Errors are displayed in two formats:

### Simple Format
```
Error: <message>
```

### Detailed Format (via `display_detailed()`)
```
<message> (<code>)
<help text if available>
```

