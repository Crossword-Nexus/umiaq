# Changelog

All notable changes to Umiaq are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-04-07

### Added
- Discord and GitHub-issues links in README and web UI

### Changed
- Replaced unmaintained `instant` crate with `web-time`

## [0.3.0] – 2026-03-28

### Added
- Negated character–set syntax: e.g., `[^AEIOU]` and `[!AEIOU]` match a letter that is NOT a vowel
- Version is now displayed in the web UI

### Fixed
- Fixed handling of `!=AB` constraints

### Changed
- Performance improvements to `check_not_equal` constraint evaluation

## [0.2.3] - 2026-03-18

### Changed
- Show version and build timestamp in error reports

### Fixed
- Miscellaneous cleanup

## [0.2.2] - 2025-02-23

### Changed
- Updated Cargo.lock to 2.3 format

## [0.2.0] - 2025-02-01

### Added
- Initial public release

[Unreleased]: https://github.com/Crossword-Nexus/umiaq/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/Crossword-Nexus/umiaq/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Crossword-Nexus/umiaq/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/Crossword-Nexus/umiaq/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Crossword-Nexus/umiaq/compare/v0.2.0...v0.2.2
[0.2.0]: https://github.com/Crossword-Nexus/umiaq/releases/tag/v0.2.0
