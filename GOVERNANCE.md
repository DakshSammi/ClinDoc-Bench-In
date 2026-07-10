# Governance

## Purpose

This repository serves both as a frozen benchmark artifact and as a living benchmark workspace for new experiments and community submissions.

## Frozen Benchmark Policy

- `benchmark_v2/final_day_freeze_20260709/` is read-only.
- Paper tables, rankings, provenance, and statistics for the frozen release are not edited in place.
- Corrections to frozen-release documentation are allowed only if they do not alter benchmark outcomes.

## What Can Change

- documentation
- scripts outside the frozen output tree
- community submission tooling
- experiment directories
- future benchmark versions released outside the frozen v1.0 tree

## Approval Expectations

Changes with the following impact should be discussed before merge:

- benchmark schema changes
- new metrics or weighting changes
- dataset expansion or replacement
- leaderboard policy changes
- privacy policy changes

## Community Results

Community results should be packaged outside the frozen benchmark and validated before comparison runs. Validation does not imply inclusion in a publication leaderboard by default.

## Privacy And Safety

- do not publish unanonymized examples
- do not commit secrets or `.env` files
- do not open public issues with patient identifiers

## Versioning Direction

- v1.0: frozen BDA 2026 release
- future versions: additive, clearly versioned, and separately frozen when published
