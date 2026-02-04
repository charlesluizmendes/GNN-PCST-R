import argparse
import json
import numpy as np
import torch
from pathlib import Path
from pcst_fast import pcst_fast as pcst
from tqdm import tqdm

def call_pcst_ensemble(edges, prizes, costs, root):
    trials = [
        (edges, prizes, costs, root),
        (edges, prizes, costs, root, 1),
        (edges, prizes, costs, root, 1, "gw"),
        (edges, prizes, costs, root, 1, "strong"),
        (edges, prizes, costs, root, 1, "gw", 0),
        (edges, prizes, costs, root, 1, "strong", 0),
    ]
    best_res, best_cost = None, float('inf')
    for args in trials:
        try:
            n, e = pcst(*args)
            c = costs[e].sum() if len(e) > 0 else 0.0
            if c < best_cost:
                best_cost, best_res = c, (n, e)
        except: pass
    
    if best_res is None: raise RuntimeError("PCST falhou")
    return best_res

def build_undirected_edges(g):
    ei, et = g["edge_index"], g["edge_type"]
    pair_type = {}
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        t = int(et[i])
        k = tuple(sorted((u, v)))

        if k not in pair_type or (t == 0 and pair_type[k] != 0):
            pair_type[k] = t
            
    edges, costs = [], []
    for (a, b), t in pair_type.items():
        edges.append([a, b])
        costs.append(1.0 if t == 0 else 1.2)
    return np.array(edges, dtype=np.int64), np.array(costs, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, nargs=2); ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    snap_dir, inst_path = Path(args.input_dir[0]), Path(args.input_dir[1]) / "instances.jsonl"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    if not inst_path.exists(): raise FileNotFoundError("instances.jsonl")
    total_lines = sum(1 for _ in open(inst_path, "r"))
    
    graph_cache, edge_cache = {}, {}
    written = 0

    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as out_f:
        for line in tqdm(open(inst_path, "r"), total=total_lines, desc="populate_labels: escrevendo labels.jsonl", unit="instância"):
            if not line.strip(): continue
            inst = json.loads(line)
            snap = inst.get("snapshot_next") or inst.get("snapshot")

            if snap not in graph_cache:
                graph_cache[snap] = torch.load(snap_dir / f"as_graph_{snap}.pt", map_location="cpu")
            g = graph_cache[snap]

            if snap not in edge_cache:
                edge_cache[snap] = build_undirected_edges(g)
            edges, costs = edge_cache[snap]

            prizes = np.zeros(g["num_nodes"]); prizes[inst["root"]] = 10.0
            for t in inst["terminals_out"]: prizes[int(t)] = 10.0

            sel_nodes, sel_e_idx = call_pcst_ensemble(edges, prizes, costs, inst["root"])
            
            inst.update({
                "tree_nodes": list(map(int, sel_nodes)),
                "tree_edges": edges[sel_e_idx].tolist()
            })
            out_f.write(json.dumps(inst, ensure_ascii=False) + "\n")
            written += 1
            
    print(f"  -> Labels gerados: {written} instâncias\n")

if __name__ == "__main__": main()