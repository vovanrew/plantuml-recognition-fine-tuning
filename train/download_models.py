#!/usr/bin/env python3
"""Download models of the ladder at their pinned commits (train/models.py) into HF_HOME.

Usage: python train/download_models.py <result json> <size> [<size> ...]
Writes, per size, the revision asked for, the revision the files came from, the bytes and seconds.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import SIZES  # noqa: E402


def main(out, sizes):
    from huggingface_hub import snapshot_download
    result = {}
    for size in sizes:
        model_id, revision = SIZES[size]
        t0 = time.time()
        folder = Path(snapshot_download(model_id, revision=revision))
        if folder.name != revision:
            raise SystemExit(f"{model_id}: got {folder.name}, pinned {revision}")
        size_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        result[size] = {"model": model_id, "revision": revision, "folder": str(folder),
                        "bytes": size_bytes, "seconds": round(time.time() - t0, 1),
                        "hf_home": os.environ.get("HF_HOME")}
        print(json.dumps({size: result[size]}), flush=True)
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
