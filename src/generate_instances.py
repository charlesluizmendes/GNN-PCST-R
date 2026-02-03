import argparse
import json
import random
from pathlib import Path
import torch
import sys

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it=None, **kwargs):
        return it

def list_graphs(input_dir):
    d = Path(input_dir)
    return sorted(d.glob("as_graph_*.pt"))

def stable_int(s):
    x = 0
    for ch in s:
        x = (x * 131 + ord(ch)) % 1000000007
    return x

def auto_instances_count(num_candidates):
    if num_candidates <= 0:
        return 0
    c = int((num_candidates ** 0.5) * 50)
    if c < 2000:
        c = 2000
    if c > 20000:
        c = 20000
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--k_terminals", type=int, default=20)
    ap.add_argument("--min_degree", type=int, default=5)
    ap.add_argument("--root_id", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_instances", type=int, default=0)
    args = ap.parse_args()

    graphs = list_graphs(args.input_dir)
    if not graphs:
        print()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "instances.jsonl"

    bar = tqdm(
        total=0,
        desc="generate_instances: escrevendo instances.jsonl",
        unit="instância",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True
    )

    total_instances = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for gp in graphs:
            g = torch.load(gp, map_location="cpu")
            n = int(g["num_nodes"])
            ei = g["edge_index"]
            snap = str(g.get("snapshot", ""))

            deg = torch.bincount(ei[0], minlength=n)
            root = args.root_id if args.root_id >= 0 else int(torch.argmax(deg).item())

            candidates = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1).tolist()
            candidates = [x for x in candidates if x != root]
            if len(candidates) == 0:
                candidates = [x for x in range(n) if x != root]

            inst_count = auto_instances_count(len(candidates))

            if args.max_instances > 0:
                remaining = args.max_instances - total_instances
                if remaining <= 0:
                    break
                if inst_count > remaining:
                    inst_count = remaining

            bar.total = int(bar.total) + int(inst_count)
            bar.refresh()

            rng = random.Random(args.seed + stable_int(snap))

            for _ in range(inst_count):
                k = args.k_terminals
                if k > len(candidates):
                    k = len(candidates)
                terminals = rng.sample(candidates, k=k)

                rec = {
                    "id": total_instances,
                    "snapshot": snap,
                    "root": root,
                    "terminals": terminals
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_instances += 1
                bar.update(1)

    bar.close()
    print()

if __name__ == "__main__":
    main()
