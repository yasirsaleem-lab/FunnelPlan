"""
Build the reproducibility package for the paper.

The package = every file tracked by git (code, docs, graded answers, audits, results)
            + the files git ignores but the analyses need: the retrieval caches data/cache_full/retrievals_v3/*.pkl
              (query text and ranked lists with scores) and the router embeddings data/processed_full/routers/emb_*.npz.
Not included (too large or regenerable, listed in ARTIFACT.md): corpora, BM25/SPLADE/dense indexes, model weights,
per call progress logs.

It writes MANIFEST.sha256 (sha256, bytes, path for every file) into the package, and the same list next to the zip
(outside the repository), so anyone can check that the package is the one the paper used.

Usage:  python scripts/make_artifact.py                 -> ../FunnelPlan_artifact_<git short hash>.zip
        python scripts/make_artifact.py --check          (list what would be packed and the total size only)
Run it on a clean tree (git status shows nothing to commit) so the hash in the file name describes the content.
"""

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXTRA = [("data/cache_full/retrievals_v3", "*.pkl"), ("data/processed_full/routers", "emb_*.npz")]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dirty = git("--no-optional-locks", "status", "--porcelain").strip()
    files = [f for f in git("ls-files").splitlines() if f and (ROOT / f).is_file()]
    for d, pat in EXTRA:
        found = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / d).glob(pat))
        if not found:
            sys.exit(f"Missing {d}/{pat}: the analyses need it. Rebuild it (see ARTIFACT.md) before packing.")
        files += found
    files = sorted(set(files))
    total = sum((ROOT / f).stat().st_size for f in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB before compression")
    if dirty:
        print("WARNING: the working tree has uncommitted changes; commit first so the package matches a commit:")
        print(dirty)
    if "--check" in sys.argv:
        big = sorted(files, key=lambda f: -(ROOT / f).stat().st_size)[:10]
        for f in big:
            print(f"  {(ROOT / f).stat().st_size / 1e6:8.1f} MB  {f}")
        return
    head = git("rev-parse", "--short", "HEAD").strip()
    lines = [f"{sha256(ROOT / f)}  {(ROOT / f).stat().st_size:>12d}  {f}" for f in files]
    manifest = "\n".join(lines) + "\n"
    out = ROOT.parent / f"FunnelPlan_artifact_{head}{'_dirty' if dirty else ''}.zip"
    out.with_suffix(".sha256.txt").write_text(manifest, encoding="utf-8")       # outside the repo: keeps the tree clean
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(ROOT / f, f"FunnelPlan/{f}")
        z.writestr("FunnelPlan/MANIFEST.sha256", manifest)
    print(f"Written {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.sha256.txt').name} (git {head})")
    print(f"sha256 of the package: {sha256(out)}")


if __name__ == "__main__":
    main()
