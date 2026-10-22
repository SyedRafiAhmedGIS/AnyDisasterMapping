# Security Policy

## Supported versions

| version | status |
|---------|--------|
| `main` (v5.x) | ✅ actively supported — report against this |
| v4.x and older | ❌ no patches; please upgrade |

## Reporting a vulnerability

**Do not open a public issue for a suspected security problem.**

1. Email **qschrx@gmail.com** with `[security]` in the subject, or use
   GitHub's *Report a vulnerability* button (Security → Advisories) if the
   repository is public.
2. Include: affected file/entry point, how to reproduce, and — if possible —
   a minimal proof of concept.
3. You will get an acknowledgement within **7 days** and a status update
   within **30 days**. We will credit reporters in the fix release unless
   you ask to stay anonymous.

## Scope — what is in bounds

Security-relevant surfaces of this codebase:

- **Checkpoint loading.** Every `torch.load` must go through
  `src/utils/safe_load.py` (`weights_only=True`). A report that any code
  path deserializes an untrusted checkpoint without that guard is in scope
  (enforced by `tests/test_repo_integrity.py::test_no_unsafe_torch_load`).
- **`tools/download_weights.py`.** Trust model: downloads from the official
  Hugging Face weights repo, verifies SHA-256 against the published
  `SHA256SUMS` / `--expect-sha256`. Weaknesses in verification, path
  handling (zip-slip), or revision pinning are in scope.
- **`tools/colab_kit.py` / notebook bootstrap.** The standalone fetch pins
  `KIT_REV` (tag) with a warned fallback to `main`. Any tampering vector in
  that path (e.g. silently executing non-repo code) is in scope.
- **Gradio demo (`gradio_demo/app.py`)** — uploaded-file handling.
- **CI workflows** — injection via untrusted input (`pull_request` targets).

## Out of scope

- Vulnerabilities in vendored third-party model code that also exist
  upstream (we will still coordinate/patch, but track them there first).
- Issues requiring a malicious dataset that the user explicitly chose to
  trust, or local attacker == user scenarios.
- The Colab "no-code demo" heuristic mode: it never loads real weights and
  is explicitly labeled `SYNTHETIC-DEMO`.

## Hardening already in place

- `weights_only=True` everywhere (tested), pickles from unknown sources
  are rejected, not executed.
- SHA-256 verification for all published weights + exact-revision pinning.
- No `eval`/`exec` on network input anywhere in `src/`, `tools/`,
  `scripts/`, `tests/` (ruff-gated).
- Dependency floor documented in `constraints.txt` with rationale
  (e.g. transformers 4.57.0 yanked; numpy 1.26 pinned for vendored code).
