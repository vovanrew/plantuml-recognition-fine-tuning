"""Tests for build/pack_runs.py, on a small synthetic runs/ folder (no stored run is read).

Run from fine-tune/:  python3 -m pytest build/tests/test_pack_runs.py -q
"""
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FT / "build"))
import pack_runs as P  # noqa: E402


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture
def base(tmp_path):
    """A runs/ folder with one test run, one dev checkpoint and two train runs."""
    r = tmp_path / "runs"
    run = r / "test" / "2B_20000"
    write(run / "answers" / "k1.puml", "@startuml\nA -> B\n@enduml\n")
    write(run / "answers" / "manifest.json", "{}")
    result = b'{"f1": 1.0}'
    write(run / "scores" / "v2" / "csr" / "2B_20000" / "csr_results.json", result)
    (run / "scores" / "v1").mkdir(parents=True)
    os.symlink("../v2/csr", run / "scores" / "v1" / "csr")
    write(run / "scores" / "manifest.json", json.dumps(
        {"files": {"v2/csr/2B_20000/csr_results.json": hashlib.sha256(result).hexdigest()}}))
    write(r / "test" / "_analysis" / "main_tables.md", "| table |")
    (r / "test" / "_root" / "v2" / "runs").mkdir(parents=True)
    os.symlink("../../../2B_20000/answers", r / "test" / "_root" / "v2" / "runs" / "2B_20000")
    write(r / "test" / ".DS_Store", "x")
    ck = r / "dev" / "2B_20000" / "step_00625"
    write(ck / "answers" / "k1.puml", "@startuml\n@enduml\n")
    write(ck / "scores" / "csr" / "csr_results.json", "{}")
    write(ck / "scores" / "csr" / "png" / "k1.png", b"\x89PNG")
    for name in ("2B_20000", "2B_1000"):
        t = r / "train" / name
        for f in P.TRAIN_FILES:
            write(t / f, f"{name} {f}")
        write(t / "state.pt", b"optimizer")
        write(t / "step_00625" / "adapter_model.safetensors", b"weights")
    return tmp_path


def test_select_takes_runs_dev_without_pngs_and_train_metadata_only(base):
    rels = P.select(base)
    assert rels == sorted(rels)
    assert "runs/test/2B_20000/answers/k1.puml" in rels
    assert "runs/test/2B_20000/scores/v1/csr" in rels                 # the link itself
    assert "runs/test/_root/v2/runs/2B_20000" in rels
    assert "runs/test/_analysis/main_tables.md" in rels
    assert "runs/dev/2B_20000/step_00625/scores/csr/csr_results.json" in rels
    assert not any("/csr/png/" in r for r in rels)
    assert not any(r.endswith(".DS_Store") for r in rels)
    train = [r for r in rels if r.startswith("runs/train/")]
    assert train == sorted(f"runs/train/{n}/{f}" for n in ("2B_1000", "2B_20000") for f in P.TRAIN_FILES)


def test_select_refuses_a_train_run_without_its_selection(base):
    (base / "runs" / "train" / "2B_1000" / "selection.json").unlink()
    with pytest.raises(P.Refused, match="selection.json"):
        P.select(base)


def test_pack_is_byte_identical_on_repeat_and_keeps_links(base, tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    P.pack(base, a)
    os.utime(base / "runs" / "test" / "_analysis" / "main_tables.md", (0, 0))   # mtime is not packed
    P.pack(base, b)
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as z:
        info = z.getinfo("runs/test/2B_20000/scores/v1/csr")
        assert stat.S_ISLNK(info.external_attr >> 16)
        assert z.read(info) == b"../v2/csr"
        assert all(i.date_time == P.DATE for i in z.infolist())


def test_pack_refuses_an_absolute_link(base, tmp_path):
    os.symlink("/etc/hosts", base / "runs" / "test" / "_analysis" / "abs")
    with pytest.raises(P.Refused, match="absolute"):
        P.pack(base, tmp_path / "x.zip")


def test_check_passes_on_a_fresh_pack(base, tmp_path):
    z = tmp_path / "r.zip"
    P.pack(base, z)
    assert P.check(base, z) == []


def test_check_finds_changed_missing_and_extra_files(base, tmp_path):
    z = tmp_path / "r.zip"
    P.pack(base, z)
    write(base / "runs" / "test" / "2B_20000" / "answers" / "k1.puml", "changed")
    write(base / "runs" / "test" / "2B_20000" / "answers" / "k2.puml", "new")
    (base / "runs" / "train" / "2B_1000" / "log.jsonl").unlink()
    write(base / "runs" / "train" / "2B_1000" / "log.jsonl", "2B_1000 log.jsonl")   # same bytes back
    problems = P.check(base, z)
    assert "bytes differ: runs/test/2B_20000/answers/k1.puml" in problems
    assert "missing from the archive: runs/test/2B_20000/answers/k2.puml" in problems
    assert not any("log.jsonl" in p for p in problems)


def test_check_reads_the_score_manifest_inside_the_archive(base, tmp_path):
    write(base / "runs" / "test" / "2B_20000" / "scores" / "manifest.json",
          json.dumps({"files": {"v2/csr/2B_20000/csr_results.json": "0" * 64}}))
    z = tmp_path / "r.zip"
    P.pack(base, z)
    assert P.check(base, z) == ["runs/test/2B_20000/scores/manifest.json: sha256 of "
                                "v2/csr/2B_20000/csr_results.json differs"]
