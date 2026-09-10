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

## [0.1.5] - 2026-09-10
### Fixed
- LaTeX equation rendering now preserves division structure with fractions,
  including products in denominators, and escapes underscores in exponent
  labels.

## [0.1.4] - 2026-07-17
### Added
- Data-driven satellite reporting through `SatelliteAnchor`,
  `SatelliteReference`, and `satellite_projection`, which converts solved
  model-volume drivers into quantities with declared units without changing
  the equilibrium system. References retain solved baseline drivers so that
  physical anchors reproduce their base point exactly and scenarios use a
  common denominator.
- `satellite_calibration_report` to retain differences between monetary
  calibration drivers and the solved satellite reference as explicit
  diagnostics.
- `SatelliteBalance` and `satellite_balances` for post-solution evaluation of
  signed quantity identities, including unit and missing-anchor checks.

## [0.1.3] - 2026-07-16
### Added
- Closure-condition roles in equation rendering, with optional role labels.
- Post-solution accounting-check residuals in `Results`, tidy exports, JSON,
  CSV, Arrow, Parquet, and DualSignals outputs.

## [0.1.2] - 2026-06-19
### Added
- Rendering of objective functions.

## [0.1.1] - 2026-05-20
### Added
- Rendering support for `JCGECore` inequality equations `ELe` and `EGe`.
- Rendering support for `JCGECore` natural logarithm expressions `ELog`.

## [0.1.0] - 2026-01-18
### Added
- JCGEOutput package layout with entry points for exporting model runs.
- Block-based output interfaces and `RunSpec` to drive report generation.
- Output writers for CSV, Arrow, and Parquet using a shared table schema.
- Calibration/output helpers aligned with JCGECore and JCGERuntime data structures.
- Example outputs and integration tests covering end-to-end export flows.
- Documentation scaffolding for package usage and output formats.
