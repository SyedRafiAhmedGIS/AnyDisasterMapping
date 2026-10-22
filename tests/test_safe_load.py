import pytest
import torch

from src.utils.safe_load import safe_torch_load


class _Evil:
    def __reduce__(self):  # would execute on a full unpickle
        return (print, ("pwned",))


def test_plain_state_dict_loads(tmp_path):
    p = tmp_path / "w.pth"
    torch.save({"w": torch.ones(2), "epoch": 3, "best": 0.5}, p)
    out = safe_torch_load(p)
    assert out["epoch"] == 3 and torch.equal(out["w"], torch.ones(2))


def test_arbitrary_object_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ANYDISASTER_TRUST_CHECKPOINTS", raising=False)
    p = tmp_path / "evil.pth"
    torch.save({"x": _Evil()}, p)
    with pytest.raises(RuntimeError, match="Refusing to unpickle"):
        safe_torch_load(p)


def test_explicit_trust_opt_in(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("ANYDISASTER_TRUST_CHECKPOINTS", "1")
    p = tmp_path / "obj.pth"
    torch.save({"x": _Evil()}, p)
    safe_torch_load(p)  # allowed only with the opt-in
