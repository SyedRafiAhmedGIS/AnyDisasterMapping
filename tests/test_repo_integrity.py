"""Cheap, torch-free guards against the kind of drift that broke the repo before:
dvc.yaml / Dockerfile / README pointing at files that do not exist."""
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _expand(text, variables):
    for k, v in variables.items():
        text = text.replace("${%s}" % k, str(v))
    return text


def test_dvc_yaml_paths_exist():
    spec = yaml.safe_load((REPO / "dvc.yaml").read_text())
    variables = {}
    for item in spec.get("vars", []):
        variables.update(item)
    assert spec["stages"], "dvc.yaml defines no stages"
    outs = set()
    for stage in spec["stages"].values():
        outs.update(_expand(o, variables) for o in stage.get("outs", []))
    for name, stage in spec["stages"].items():
        for dep in stage.get("deps", []):
            dep = _expand(dep, variables)
            # deps produced by an earlier stage (data/run dirs) need not exist yet
            if dep in outs or any(dep.startswith(o.rstrip("/") + "/") for o in outs):
                continue
            assert (REPO / dep).exists(), f"dvc stage '{name}': missing dep {dep}"
        cmd = _expand(stage["cmd"], variables)
        for token in cmd.split():
            if token.endswith(".py") and "/" in token:
                assert (REPO / token).exists(), f"dvc stage '{name}': missing script {token}"
            if token.endswith(".yaml"):
                assert (REPO / token).exists(), f"dvc stage '{name}': missing config {token}"


def test_dockerfile_copy_sources_exist():
    for line in (REPO / "Dockerfile").read_text().splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(.+)\s+\S+\s*$", line)
        if not m:
            continue
        for src in m.group(1).split():
            if src.startswith("--"):
                continue
            assert (REPO / src).exists(), f"Dockerfile COPY source missing: {src}"


def test_demo_binds_for_docker():
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "GRADIO_SERVER_NAME=0.0.0.0" in dockerfile


def test_readme_notebooks_listed_and_present():
    readme = (REPO / "README.md").read_text()
    notebooks = sorted(p.name for p in REPO.glob("*.ipynb"))
    for nb in notebooks:
        assert f"`{nb}`" in readme, f"{nb} is not listed in README"
    m = re.search(r"collection \((\d+) notebooks\)", readme)
    assert m and int(m.group(1)) == len(notebooks)


def test_license_files_present():
    for f in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert (REPO / f).is_file(), f"missing {f}"
    assert any((REPO / "third_party_licenses").glob("DINOv3*"))


def test_no_unsafe_torch_load():
    """Every torch.load must go through safe_torch_load (weights_only=True)."""
    offenders = []
    for base in ("src", "tools", "scripts", "gradio_demo"):
        for p in (REPO / base).rglob("*.py"):
            if p.name == "safe_load.py" or "kernels" in p.parts:
                continue
            for i, line in enumerate(p.read_text(encoding="utf8").splitlines(), 1):
                s = line.strip()
                if s.startswith("#"):
                    continue
                if re.search(r"(?<![\w.])torch\.load\(", s) and "weights_only=True" not in s:
                    offenders.append(f"{p.relative_to(REPO)}:{i}: {s[:80]}")
    # profiling.py patches torch.load itself and SatMAE/SpectralGPT use a guarded
    # wrapper that tries weights_only=True first.
    allowed = ("src/utils/profiling.py", "src/models/SatMAE/SatMAEDPT.py",
               "src/models/SpectralGPT/SpectralGPT.py")
    offenders = [o for o in offenders if not o.startswith(allowed)]
    assert not offenders, "unsafe torch.load calls:\n" + "\n".join(offenders)
