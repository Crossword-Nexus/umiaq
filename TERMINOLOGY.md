# umiaq terminology
* **input**: a semicolon (`;`)–separated list of **input item**s
* **output**: a list of **patterns instantiation**s, one per line
* **input item**: an **(entry) pattern** or a **constraint**
* **patterns instantiation**: a bullet (`•`)-separated list of **pattern instantiation**s
* **(entry) pattern**: a concatenated list of **(entry) pattern character**s
* **constraint**: one of a **length equality**, a **length inequality**, a **complex constraint**
* **pattern instantiation**: a concatenated list of lowercase letters
* **(entry) pattern character**: one of a **variable**, a lowercase letter, a **wildcard**, a **reversed variable**, an **anagram expression**
* **length equality**: concatenation of a pipe (`|`), an **(entry) pattern**, a pipe (`|`), an equal sign (`=`), and a nonzero number of digits (not beginning with 0)
* **length inequality**: concatenation of a pipe (`|`), an **(entry) pattern**, a pipe (`|`), an **inequality operator**, and a nonzero number of digits (not beginning with 0)
* **complex constraint**: concatenation of a **variable**, an equal sign (`=`), a left parenthesis (`(`), a **complex-constraint core**, a right parenthesis (`)`)
* **variable**: a capital English letter
* **wildcard**: one of a `.`, `*`, `@`, `#`, **character class**
* **reversed variable**: a `~` followed by a **variable**
* **anagram expression**: a `/` followed by lowercase letters
* **inequality operator**: one of `!=`, `<=`, `>=`, `<`, `>`
* **complex-constraint core**: one of a **length-regex constraint pair**, a **length constraint**, a **regex constraint**
* **character class**: concatenation of a left bracket (`[`), a **character-class core**, and a right bracket (`]`)
* **length-regex constraint pair**: concatenation of a **length constraint**, a colon (`:`), a **regex constraint**
* **length constraint**: concatenation of any number of digits (including an empty string; not beginning with 0), a hyphen (`-`), any number of digits (including an empty string; not beginning with 0)
* **regex constraint**†: concatenation of any number of **regex character**s
* **character-class core**: a concatenation of **simple character-class core**s
* **regex character**†: one of a lowercase letter, a **wildcard**
* **simple character-class core**: a lowercase letter or a **character-class range**
* **character-class range**: concatenation of a lowercase letter, a hyphen (`–`), and a lowercase letter

†only the regular expressions described here are supported
