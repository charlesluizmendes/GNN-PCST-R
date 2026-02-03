import argparse
import json
import numpy as np
import torch
from pathlib import Path
from pcst_fast import pcst_fast as pcst
from tqdm import tqdm

def call_pcst(fn, edges, prizes, costs, root):
    edges = np.asarray(edges, dtype=np.int64)
    prizes = np.asarray(prizes, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    root = int(root)

    trials = [
        (edges, prizes, costs, root),
        (edges, prizes, costs, root, 1),
        (edges, prizes, costs, root, 1, "gw"),
        (edges, prizes, costs, root, 1, "strong"),
        (edges, prizes, costs, root, 1, "gw", 0),
        (edges, prizes, costs, root, 1, "strong", 0),
    ]
    last = None
    for t in trials:
        try:
            return fn(*t)
        except Exception as e:
            last = e
    raise last

def load_graph(path):
    return torch.load(path, map_location="cpu")

def build_undirected_edges(g):
    ei = g["edge_index"]
    et = g["edge_type"]
    n_edges = int(ei.size(1))

    pair_type = {}
    for i in range(n_edges):
        u = int(ei[0, i].item())
        v = int(ei[1, i].item())
        t = int(et[i].item())
        a = u if u < v else v
        b = v if u < v else u
        key = (a, b)
        prev = pair_type.get(key)
        if prev is None:
            pair_type[key] = t
        else:
            if prev != 0 and t == 0:
                pair_type[key] = 0

    edges = []
    costs = []
    for (a, b), t in pair_type.items():
        edges.append([a, b])
        if t == 0:
            costs.append(0.1) 
        else:
            costs.append(0.5) 

    edges_np = np.asarray(edges, dtype=np.int64)
    costs_np = np.asarray(costs, dtype=np.float64)
    return edges_np, costs_np

def iter_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, nargs=2)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    snapshots_dir = Path(args.input_dir[0])
    instances_dir = Path(args.input_dir[1])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_path = instances_dir / "instances.jsonl"
    if not instances_path.exists():
        raise FileNotFoundError(f"instances.jsonl não encontrado em {instances_dir}")

    total = count_lines(instances_path)

    graph_cache = {}
    edge_cache = {}

    out_path = out_dir / "labels.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for inst in tqdm(
            iter_instances(instances_path),
            total=total,
            desc="populate_labels: escrevendo labels.jsonl",
            unit="instância"
        ):
            snap_next = inst.get("snapshot_next") or inst.get("snapshot")
            gid = f"as_graph_{snap_next}.pt"
            gp = snapshots_dir / gid

            if snap_next not in graph_cache:
                if not gp.exists():
                    raise FileNotFoundError(f"Grafo não encontrado: {gp}")
                g = load_graph(gp)
                graph_cache[snap_next] = g
            else:
                g = graph_cache[snap_next]

            n = int(g["num_nodes"])

            if snap_next not in edge_cache:
                edges_u, costs_u = build_undirected_edges(g)
                edge_cache[snap_next] = (edges_u, costs_u)
            else:
                edges_u, costs_u = edge_cache[snap_next]

            prizes = np.zeros(n, dtype=np.float64)
            root = int(inst["root"])

            terminals = inst.get("terminals_out")
            if terminals is None:
                terminals = inst.get("terminals", [])

            terminals = [int(x) for x in terminals]

            prizes[root] = 10.0 
            for t in terminals:
                prizes[t] = 10.0

            sel_nodes, sel_edges = call_pcst(pcst, edges_u, prizes, costs_u, root)

            sel_nodes = [int(x) for x in np.asarray(sel_nodes, dtype=np.int64).tolist()]
            sel_edges = [int(x) for x in np.asarray(sel_edges, dtype=np.int64).tolist()]

            tree_edges = []
            for ei_idx in sel_edges:
                a = int(edges_u[ei_idx][0])
                b = int(edges_u[ei_idx][1])
                tree_edges.append([a, b])

            rec = {
                "id": int(inst["id"]),
                "snapshot": inst.get("snapshot", ""),
                "snapshot_next": snap_next,
                "history_snapshots": inst.get("history_snapshots", []), 
                "root": root,
                "interest_id": int(inst.get("interest_id", -1)),
                "radius_hops": int(inst.get("radius_hops", -1)),
                "terminals_in": inst.get("terminals_in", []),
                "terminals_out": terminals,
                "tree_nodes": sel_nodes,
                "tree_edges": tree_edges
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"  -> Labels gerados: {written} instâncias")
    print()

if __name__ == "__main__":
    main()