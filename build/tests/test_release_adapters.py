"""Tests for build/release_adapters.py, on synthetic runs (no stored adapter is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_release_adapters.py -q
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import release_adapters as R  # noqa: E402

REV = {"Qwen/Qwen3.5-2B": "a" * 40, "Qwen/Qwen3.5-9B": "b" * 40, "Qwen/Qwen3.5-27B": "c" * 40}


def sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def base(tmp_path):
    """runs/train and runs/test for the five released runs, with tiny adapter files."""
    for repo, (model, adapters) in R.REPOS.items():
        for run, _, _ in adapters:
            step = "step_00126" if run == "2B_1000" else "step_02500"
            folder = tmp_path / "runs" / "train" / run / step
            folder.mkdir(parents=True)
            files = {"adapter_model.safetensors": f"weights {run}".encode(),
                     "adapter_config.json": f'{{"run": "{run}"}}'.encode()}
            for f, data in files.items():
                (folder / f).write_bytes(data)
            # a checkpoint that was not selected must not be released
            (tmp_path / "runs" / "train" / run / "step_00032").mkdir()
            (tmp_path / "runs" / "train" / run / "run.json").write_text(
                json.dumps({"model": model, "revision": REV[model]}))
            test = tmp_path / "runs" / "test" / run
            test.mkdir(parents=True)
            adapter = {"path": f"/workspace/test/fine-tune/runs/train/{run}/{step}"}
            adapter.update({f: sha(d) for f, d in files.items()})
            (test / "adapter_check.json").write_text(json.dumps(
                {"identity": {"model": model, "revision": REV[model], "adapter": adapter}}))
    return tmp_path


def test_served_reads_the_checkpoint_the_test_run_served(base):
    s = R.served("2B_1000", base)
    assert s["step"] == "step_00126"
    assert s["folder"] == base / "runs" / "train" / "2B_1000" / "step_00126"
    assert s["revision"] == REV["Qwen/Qwen3.5-2B"]


def test_served_refuses_a_revision_the_run_was_not_trained_on(base):
    p = base / "runs" / "train" / "9B_20000" / "run.json"
    p.write_text(json.dumps({"model": "Qwen/Qwen3.5-9B", "revision": "d" * 40}))
    with pytest.raises(R.Refused, match="served"):
        R.served("9B_20000", base)


def test_card_names_base_revision_licence_and_every_adapter(base):
    c = R.card("Qwen3.5-2B-PlantUML-LoRA", base)
    assert "base_model: Qwen/Qwen3.5-2B\n" in c
    assert "license: apache-2.0\n" in c and "library_name: peft\n" in c
    assert c.count(REV["Qwen/Qwen3.5-2B"]) == 3                     # text, base load, processor
    assert 'subfolder="pairs-20000"' in c and '"vovanrew/Qwen3.5-2B-PlantUML-LoRA"' in c
    for run, sub, pairs in R.REPOS["Qwen3.5-2B-PlantUML-LoRA"][1]:
        assert f"| `{sub}` | {pairs:,} |" in c
        assert sha(f"weights {run}".encode()) in c
    assert "$" not in c.replace("$ ", "")                          # no placeholder left
    one = R.card("Qwen3.5-9B-PlantUML-LoRA", base)
    assert "LoRA adapter for" in one and "It is the fine-tuned model" in one


def test_stage_writes_only_the_released_files_and_checks_them(base, tmp_path):
    out = tmp_path / "stage"
    assert R.stage(out, base) == []
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    want = sorted([f"{repo}/{n}" for repo in R.REPOS for n in ("README.md", "LICENSE")]
                  + [f"{repo}/{sub}/{f}" for repo, (_, ads) in R.REPOS.items()
                     for _, sub, _ in ads for f in R.FILES])
    assert files == want
    assert (out / "Qwen3.5-27B-PlantUML-LoRA" / "LICENSE").read_text().lstrip().startswith("Apache License")


def test_stage_refuses_an_existing_folder(base, tmp_path):
    (tmp_path / "stage" / "Qwen3.5-2B-PlantUML-LoRA").mkdir(parents=True)
    with pytest.raises(R.Refused, match="exists"):
        R.stage(tmp_path / "stage", base)


def test_check_finds_a_changed_adapter_and_a_stray_file(base, tmp_path):
    out = tmp_path / "stage"
    R.stage(out, base)
    (out / "Qwen3.5-2B-PlantUML-LoRA" / "pairs-5000" / "adapter_model.safetensors").write_bytes(b"other")
    (out / "Qwen3.5-9B-PlantUML-LoRA" / "state.pt").write_bytes(b"x")
    problems = R.check(out, base)
    assert "Qwen3.5-2B-PlantUML-LoRA/pairs-5000/adapter_model.safetensors: sha256 is not the one 2B_5000 served" in problems
    assert "Qwen3.5-9B-PlantUML-LoRA: not released state.pt" in problems


class FakeApi:
    """model_info as the Hub returns it, from a staged folder; `lfs.sha256` for the weights."""
    def __init__(self, out):
        self.out = out

    def model_info(self, repo_id, files_metadata=False):
        d = self.out / repo_id.split("/")[1]
        sib = []
        for p in sorted(d.rglob("*")):
            if p.is_file():
                rel = p.relative_to(d).as_posix()
                lfs = SimpleNamespace(sha256=R.sha256_file(p)) if p.suffix == ".safetensors" else None
                sib.append(SimpleNamespace(rfilename=rel, lfs=lfs))
        return SimpleNamespace(siblings=sib, sha="0" * 40)


def test_check_hub_compares_the_hub_sha256_with_the_served_one(base, tmp_path, monkeypatch):
    out = tmp_path / "stage"
    R.stage(out, base)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, revision: str(out / repo_id.split("/")[1] / filename))
    assert R.check_hub(base, FakeApi(out)) == []
    (out / "Qwen3.5-27B-PlantUML-LoRA" / "pairs-20000" / "adapter_model.safetensors").write_bytes(b"other")
    (out / "Qwen3.5-9B-PlantUML-LoRA" / "LICENSE").unlink()
    problems = R.check_hub(base, FakeApi(out))
    assert any("Qwen3.5-27B-PlantUML-LoRA/pairs-20000/adapter_model.safetensors" in p for p in problems)
    assert "vovanrew/Qwen3.5-9B-PlantUML-LoRA: missing LICENSE" in problems
