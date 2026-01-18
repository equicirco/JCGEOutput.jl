# JCGEOutput Changelog
All notable changes to this project will be documented in this file.
Releases use semantic versioning as in 'MAJOR.MINOR.PATCH'.

## Change entries
Added: For new features that have been added.
Changed: For changes in existing functionality.
Deprecated: For once-stable features removed in upcoming releases.
Removed: For features removed in this release.
Fixed: For any bug fixes.
Security: For vulnerabilities.

## [0.1.0] - 2026-01-18
### Added
- JCGEOutput package layout with entry points for exporting model runs.
- Block-based output interfaces and `RunSpec` to drive report generation.
- Output writers for CSV, Arrow, and Parquet using a shared table schema.
- Calibration/output helpers aligned with JCGECore and JCGERuntime data structures.
- Example outputs and integration tests covering end-to-end export flows.
- Documentation scaffolding for package usage and output formats.
