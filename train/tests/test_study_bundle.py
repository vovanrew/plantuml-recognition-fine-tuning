"""Tests for train/study_bundle.sh, on throwaway git repositories and on this one.

Run from fine-tune/:
  .venv/bin/python train/tests/test_study_bundle.py
(or with pytest: .venv/bin/python -m pytest <this file>)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

FT = Path(__file__).resolve().parents[2]
SCRIPT = FT / "train" / "study_bundle.sh"
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", "-c", "advice.detachedHead=false"]


def sh(*args, cwd=None):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], cwd=cwd, capture_output=True, text=True)


def git(repo, *args):
    return subprocess.run([*GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def repository(tmp, tagged=True):
    """A repository with infer/manifest.py naming serving-v3, one commit, the tag if `tagged`."""
    repo = Path(tmp) / "study"
    (repo / "infer").mkdir(parents=True)
    (repo / "infer" / "manifest.py").write_text('FREEZE_TAG = "serving-v3"\n')
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "one")
    if tagged:
        git(repo, "tag", "-a", "serving-v3", "-m", "frozen")
    return repo


def clone(bundle, into):
    git(Path(bundle).parent, "clone", "-q", str(bundle), str(into))
    return into


def test_create_refuses_a_repository_without_the_freeze_tag():
    with tempfile.TemporaryDirectory() as tmp:
        repo = repository(tmp, tagged=False)
        done = sh("create", repo, Path(tmp) / "study.bundle")
        assert done.returncode == 1 and "serving-v3 is not in" in done.stderr, done.stdout + done.stderr
        assert not (Path(tmp) / "study.bundle").exists()


def test_the_bundle_carries_the_tag_and_its_clone_passes_the_check():
    with tempfile.TemporaryDirectory() as tmp:
        repo = repository(tmp)
        done = sh("create", repo, Path(tmp) / "study.bundle")
        assert done.returncode == 0, done.stdout + done.stderr
        checkout = clone(Path(tmp) / "study.bundle", Path(tmp) / "pod")
        git(checkout, "checkout", "-q", git(repo, "rev-parse", "HEAD"))
        assert git(checkout, "rev-parse", "refs/tags/serving-v3") == git(repo, "rev-parse", "refs/tags/serving-v3")
        done = sh("check", checkout)
        assert done.returncode == 0 and "serving-v3 present" in done.stdout


def test_a_relative_bundle_path_is_taken_from_where_the_command_runs():
    with tempfile.TemporaryDirectory() as tmp:
        repo = repository(tmp)
        (Path(tmp) / "pack").mkdir()
        done = sh("create", repo, "pack/study.bundle", cwd=tmp)
        assert done.returncode == 0, done.stdout + done.stderr
        assert (Path(tmp) / "pack" / "study.bundle").is_file() and not (repo / "pack").exists()


def test_a_bundle_of_head_alone_fails_the_check():
    """The pilot session's pack (HEAD only): its checkout had no tag."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = repository(tmp)
        git(repo, "bundle", "create", str(Path(tmp) / "head.bundle"), "HEAD")
        checkout = clone(Path(tmp) / "head.bundle", Path(tmp) / "pod")
        done = sh("check", checkout)
        assert done.returncode == 1 and "serving-v3 is not in" in done.stderr, done.stdout + done.stderr


def test_this_repository_bundles_with_its_freeze_tag():
    with tempfile.TemporaryDirectory() as tmp:
        done = sh("create", FT, Path(tmp) / "study.bundle")
        assert done.returncode == 0, done.stdout + done.stderr
        heads = subprocess.run(["git", "bundle", "list-heads", str(Path(tmp) / "study.bundle")],
                               capture_output=True, text=True, check=True).stdout
        assert f"{git(FT, 'rev-parse', 'refs/tags/serving-v3')} refs/tags/serving-v3" in heads


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name, flush=True)
        except BaseException as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, f"{type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
