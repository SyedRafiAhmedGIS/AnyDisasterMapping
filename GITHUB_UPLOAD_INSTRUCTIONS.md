# GitHub Upload Instructions — AnyDisasterMapping-extended

Two ways to publish this project. Do **one** of them, then verify the checklist at the end.

---

## Option A — Git command line (recommended)

### 1. Create the GitHub repository
1. Go to <https://github.com/new>.
2. Repository name: `AnyDisasterMapping` (or keep your fork's name).
3. Choose **Public** (needed for the Colab "Open in Colab" badge to work).
4. **Do NOT** initialize with README/.gitignore/license (you already have them).

### 2. Push this folder

```bash
# unzip if you haven't already
unzip AnyDisasterMapping-extended.zip -d adm && cd adm/AnyDisasterMapping-extended

git init
git checkout -b main

# (Recommended) create a branch for the extension
git checkout -b extended-domains

git add .
git commit -m "Extended: 9 disaster domains, Colab demo, smoke tests, docs, illustrations

- 9 dataset adapters + shared helper (seg/cd/scd contracts, 255=ignore)
- 20 configs incl. first shipped task:scd config (GLOF)
- make_dummy.py synthetic generators + sanity_check.py + smoke_test_all_domains.py
- demo_colab.ipynb: auto-setup, upload widgets w/ preview, 3 honest modes, auto-export
- AI-generated / programmatic assets only (assets/CREDITS.md)
- README overhaul, ADDITIONAL_DISASTERS catalog (21 candidates), Apache-2.0 LICENSE"

git remote add origin https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping.git
git push -u origin extended-domains        # push the extension branch
git checkout main && git merge extended-domains && git push -u origin main
```

> Replace `SyedRafiAhmedGIS` with your actual username (also see step 3).
> If your upstream is a fork of the original repo, add it as `upstream` instead
> of `origin` and push your branch: `git remote add upstream <original-url>`.

### 3. Fix the Colab button URL
The README badge and `demo_colab.ipynb` first cell contain:

```
https://colab.research.google.com/github/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/demo_colab.ipynb
```

After pushing, replace **SyedRafiAhmedGIS** with your username in:
- `README.md` (badge, line ~8)
- `demo_colab.ipynb` (REPO_URL in the setup cell + badge in cell 1)

Quick sed:
```bash
sed -i 's|SyedRafiAhmedGIS|<your-username>|g' README.md demo_colab.ipynb
git add README.md demo_colab.ipynb && git commit -m "Set Colab badge username" && git push
```

### 4. GitHub Actions CI — already shipped
This repository now carries its own workflows (Round 5):

- `.github/workflows/ci.yml` — ruff lint + notebook validator + syntax/imports
  + unit tests + one-domain smoke, matrixed over torch 2.5.1 and 2.14.1;
- `.github/workflows/docs.yml` — deploys the mkdocs site to GitHub Pages
  (enable **Settings → Pages → Source: gh-pages** once, after the first push);
- `.github/workflows/release.yml` — builds sdist/wheel + SHA256SUMS and
  attaches them to a GitHub Release whenever a `v*` tag is pushed.

No extra setup needed — they activate on the first push.

### 5. (Required for the Colab kit pin) create the release tag
The 17 notebooks fetch `tools/colab_kit.py` pinned to **`v5.0.0`** when run
standalone in Colab. The pin falls back to `main` with a visible warning
until the tag exists, so push it right after the first upload:

```bash
git tag v5.0.0
git push origin v5.0.0        # also triggers the release workflow
```

When you later change `tools/colab_kit.py`, bump **both** `KIT_REV` in the
kit and the bootstrap pin inside every `*_colab.ipynb` (grep for
`for _rev in (`), and tag the next version. `scripts/validate_notebooks.py`
and `tests/test_colab_kit.py::test_kit_rev_pin_consistent_across_notebooks`
enforce consistency.

---

## Option B — GitHub web UI (drag & drop, no git needed)

1. Open <https://github.com/new> → create the repo (public, empty).
2. On the repo page click **"uploading an existing file"** (or *Add file ▸ Upload files*).
3. **Drag the CONTENTS** of `AnyDisasterMapping-extended/` (not the zip itself —
   GitHub rejects >100 files per drag in some browsers, so drag in 2–3 batches:
   `src/`+`configs/`, then `scripts/`+`assets/`+`docs/`, then root files).
4. Commit message: `Extended: 9 disaster domains, Colab demo, docs, illustrations`.
5. Click **Commit changes**.
6. Then fix the Colab badge: open `README.md` on github.com → pencil icon →
   replace `SyedRafiAhmedGIS` → commit.

> Web-UI caveat: uploading the `demo_colab.ipynb` and README this way is fine,
> but later edits are easier via git. Drags with 600+ config files may need
> splitting; `configs/` is the big folder.

---

## Post-upload checklist

- [ ] Repo is **public** (Colab badge only works for public repos)
- [ ] `README.md` renders with hero image, badges, and domain table
- [ ] Colab badge URL contains your real username
- [ ] `LICENSE` shows the full Apache-2.0 text (already replaced — just verify)
- [ ] Open the repo in Colab via the badge → `Runtime ▸ Run all` runs to the export cell
- [ ] `.gitignore` excludes `data/`, `results/`, `pretrained_weight/`, `__pycache__/` (already configured)
- [ ] No file in the repo exceeds GitHub's 100 MB hard limit (largest asset ≈ 0.3 MB — safe)

## Local pre-flight (before pushing)

```bash
unzip AnyDisasterMapping-extended.zip -d adm && cd adm/AnyDisasterMapping-extended
pip install -e .
python train.py --help && python test.py --help
python scripts/smoke_test_all_domains.py            # all 12 runs pass
python scripts/data_prep/sanity_check.py --list
```
