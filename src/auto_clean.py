import argparse
import sys
from pathlib import Path

def _tqdm(iterable, total=None, desc=None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    args = ap.parse_args()

    root = Path(args.input_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"input_dir invalido: {root}")

    exts = {".csv", ".txt", ".pt", ".jsonl"}
    targets = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            targets.append(p)

    removed = 0
    failed = 0

    for p in _tqdm(targets, total=len(targets), desc="auto_clean: removendo .csv .txt .pt .jsonl"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            failed += 1

    if failed > 0:
        sys.exit(1)
    
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)
