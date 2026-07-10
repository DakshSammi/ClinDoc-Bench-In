# Community Submissions

This directory is for benchmark submissions that should not modify the frozen benchmark release.

Use one subdirectory per submission:

```text
community/submissions/<submission_name>/
```

Start from:

- [community/submissions/template](submissions/template)
- [docs/submitting_results.md](../docs/submitting_results.md)

Validate a submission with:

```bash
python scripts/validate_submission.py --submission-dir community/submissions/your_submission
```
