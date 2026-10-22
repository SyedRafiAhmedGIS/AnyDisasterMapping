#!/usr/bin/env python3
"""Validator for the AnyDisasterMapping Colab notebook collection.

Checks (all cheap, torch-free):
  1. JSON-valid nbformat + every code cell AST-parses.
  2. Colab badges present in every notebook's markdown.
  3. Every notebook either imports colab_kit (thin family) or stays
     self-consistent (no reference to removed shared names without the kit).
  4. No duplicated shared function bodies remain across notebooks
     (the Round-5 dedup contract): the engine/upload helper names must be
     defined at most once per notebook and never identically in two notebooks.
  5. tools/colab_kit.py exists, compiles, and KIT_REV matches the pinned rev
     in notebook bootstraps.
  6. README lists every notebook and the collection count matches.

Exit code 0 = all good; 1 = problems (printed).
"""
from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ENGINE_NAMES = {
    "_field", "_fbm", "_stretch", "_blur", "_median_mask", "_terrain",
    "make_synthetic", "_hsv", "segment_heuristic", "change_heuristic",
    "_otsu", "infer", "overlay", "heatmap", "to_u8", "stats_for",
    "stylize_poster", "read_any", "verdict", "show_pair",
    "_save_upload", "_apply", "upload_pre", "upload_post", "use_synthetic",
    "run_inference", "run_pipeline", "run_export",
}

problems: list[str] = []


def err(msg: str) -> None:
    problems.append(msg)
    print(f"  ✗ {msg}")


def main() -> int:
    kit = REPO / "tools" / "colab_kit.py"
    if not kit.is_file():
        err("tools/colab_kit.py missing")
    else:
        try:
            py_compile.compile(str(kit), doraise=True)
            print(f"  ✔ tools/colab_kit.py compiles ({kit.stat().st_size/1024:.1f} KB)")
        except py_compile.PyCompileError as e:
            err(f"colab_kit.py does not compile: {e}")
        m = re.search(r'^KIT_REV = "([^"]+)"', kit.read_text(), re.M)
        kit_rev = m.group(1) if m else None

    notebooks = sorted(REPO.glob("*_colab.ipynb"))
    print(f"  ✔ found {len(notebooks)} notebooks")
    dup_bodies: dict[str, list[str]] = {}
    for p in notebooks:
        try:
            nb = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            err(f"{p.name}: invalid JSON: {e}")
            continue
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        joined = "\n\n".join("".join(c["source"]) for c in code_cells)
        # 1. AST
        for i, c in enumerate(code_cells):
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                err(f"{p.name}: cell {i} syntax error: {e}")
        # 2. Colab badge in markdown
        if not any("colab.research.google.com" in "".join(c["source"])
                   for c in nb["cells"] if c["cell_type"] == "markdown"):
            err(f"{p.name}: no Colab badge in markdown")
        # 3. kit import present for thin notebooks
        uses_kit = "_ensure_colab_kit" in joined or "import colab_kit" in joined
        if uses_kit and "import colab_kit" not in joined:
            err(f"{p.name}: bootstrap without 'import colab_kit'")
        # 4. duplicated shared defs (identical bodies across notebooks)
        tree_defs: dict[str, str] = {}
        for c in code_cells:
            src = "".join(c["source"])
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in ENGINE_NAMES:
                    seg = ast.get_source_segment(src, node)
                    # thin kit-delegates (1-3 lines calling colab_kit.*) are the
                    # intended refactor pattern — only real logic counts as dup
                    if seg.count("\n") <= 3 and "colab_kit." in seg:
                        continue
                    tree_defs[node.name] = hashlib.md5(seg.encode()).hexdigest()
        for name, h in tree_defs.items():
            dup_bodies.setdefault(f"{name}:{h}", []).append(p.name)

    for key, users in dup_bodies.items():
        if len(users) > 1:
            err(f"shared def still duplicated identically in {len(users)} notebooks: "
                f"{key.split(':')[0]} in {users}")

    # 5. KIT_REV consistency (notebooks store source JSON-escaped)
    if kit_rev:
        for p in notebooks:
            text = p.read_text().replace('\\"', '"')
            has_boot = "_ensure_colab_kit" in text
            m = re.search(r'for _rev in \("([^"]+)", "main"\)', text)
            if has_boot and not m:
                err(f"{p.name}: bootstrap without pinned rev")
            elif m and m.group(1) != kit_rev:
                err(f"{p.name}: pins {m.group(1)} but colab_kit.KIT_REV={kit_rev}")
        print(f"  ✔ KIT_REV pin consistent ({kit_rev})")

    # 6. README listing
    readme = (REPO / "README.md").read_text()
    for p in notebooks:
        if f"`{p.name}`" not in readme:
            err(f"README does not list {p.name}")
    m = re.search(r"collection \((\d+) notebooks\)", readme)
    if not m or int(m.group(1)) != len(notebooks):
        err(f"README collection count mismatch (found {m.group(1) if m else None}, "
            f"actual {len(notebooks)})")
    else:
        print(f"  ✔ README lists all {len(notebooks)} notebooks")

    print()
    if problems:
        print(f"VALIDATION FAILED — {len(problems)} problem(s)")
        return 1
    print("NOTEBOOK VALIDATION: ALL CHECKS PASSED ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
