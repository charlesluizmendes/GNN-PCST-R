import argparse
import csv
import json
import heapq
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def _find_file(root: Path, name: str):
    direct = root / name
    if direct.exists(): return direct
    matches = list(root.rglob(name))
    return matches[0] if matches else None

def _load_nodes(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = next((c for c in reader.fieldnames if c.lower() in ["node_id", "id", "asn"]), reader.fieldnames[0])
        ids = [int(row[col]) for row in reader if row[col].strip()]
        return set(ids)

def _dijkstra(adj, root, terminals, cost_map):
    dist = [float("inf")] * len(adj)
    prev = [-1] * len(adj)
    dist[root], heap = 0.0, [(0.0, root)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v], prev[v] = d + w, u
                heapq.heappush(heap, (dist[v], v))
    
    edges = set()
    for t in terminals:
        curr = t
        while curr != root and curr != -1:
            p = prev[curr]
            if p == -1: break
            edges.add(tuple(sorted((p, curr))))
            curr = p
            
    return sum(cost_map.get(e, 0) for e in edges)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()
    root_dir = Path(args.input_dir)

    nodes_csv = _find_file(root_dir, "nodes.csv")
    snaps_txt = _find_file(root_dir, "snapshots.txt")
    inst_jsonl = _find_file(root_dir, "instances.jsonl")
    lab_jsonl = _find_file(root_dir, "labels.jsonl")

    node_set = _load_nodes(nodes_csv)
    snap_ids = [l.strip() for l in snaps_txt.read_text().splitlines() if l.strip()]
    instances = [json.loads(l) for l in inst_jsonl.read_text().splitlines() if l.strip()]
    labels = [json.loads(l) for l in lab_jsonl.read_text().splitlines() if l.strip()]

    graphs = {}
    total_n, total_e = 0, 0
    for sid in tqdm(snap_ids, desc="validate_snapshots"):
        p = _find_file(root_dir, f"as_graph_{sid}.pt") or _find_file(root_dir, f"{sid}.pt")
        if not p:
            matches = list(root_dir.glob(f"*{sid}*.pt"))
            if matches: p = matches[0]
            else: raise FileNotFoundError(f"Não foi possível encontrar o grafo para o snapshot {sid}")
            
        g = torch.load(p, map_location="cpu")
        graphs[sid] = g
        total_n += g.get("num_nodes", 0)
        total_e += g["edge_index"].size(1)

    st = {"conn": 0, "hits": 0, "total_edges": 0, "nodes": [], "edges": [], "saving": [], "steiner": 0}
    
    for lab in tqdm(labels, desc="validate_labels"):
        snap_key = lab["snapshot_next"]
        g = graphs[snap_key]
        num_nodes = int(g.get("num_nodes", len(node_set)))
        
        ei, et = g["edge_index"], g.get("edge_type", torch.zeros(g["edge_index"].size(1)))
        cost_map, adj = {}, [[] for _ in range(num_nodes)]
        for i in range(ei.size(1)):
            u, v, t = int(ei[0, i]), int(ei[1, i]), int(et[i])
            pair = tuple(sorted((u, v)))
            c = 1.0 if t == 0 else 1.2
            if pair not in cost_map or c < cost_map[pair]:
                cost_map[pair] = c
                adj[u].append((v, c)); adj[v].append((u, c))

        pcst_cost = 0.0
        for e in lab["tree_edges"]:
            st["total_edges"] += 1
            pair = tuple(sorted(e))
            if pair in cost_map:
                st["hits"] += 1
                pcst_cost += cost_map[pair]

        adj_tree = {n: [] for n in lab["tree_nodes"]}
        for u, v in lab["tree_edges"]:
            adj_tree[u].append(v); adj_tree[v].append(u)
        
        root = lab["root"]
        q, seen = [root], {root}
        while q:
            u = q.pop(0)
            for v in adj_tree.get(u, []):
                if v not in seen:
                    seen.add(v); q.append(v)
        
        if seen == set(lab["tree_nodes"]) and all(t in seen for t in lab["terminals_out"]):
            st["conn"] += 1

        base_cost = _dijkstra(adj, root, lab["terminals_out"], cost_map)
        st["saving"].append(1.0 - (pcst_cost / base_cost) if base_cost > 0 else 0.0)
        st["nodes"].append(len(lab["tree_nodes"]))
        st["edges"].append(len(lab["tree_edges"]))
        st["steiner"] += (len(lab["tree_nodes"]) - len(lab["terminals_out"]) - 1)

    report = {
        "input_dir": str(root_dir),
        "nodes_count": len(node_set),
        "graphs_found": len(list(root_dir.rglob("*.pt"))),
        "graphs_checked": len(snap_ids),
        "avg_nodes_per_graph_checked": total_n / len(snap_ids),
        "avg_edges_per_graph_checked": total_e / len(snap_ids),
        "instances_found": len(instances),
        "instances_checked": len(labels),
        "labels_tree_connected": st["conn"] / len(labels),
        "labels_physical_edge_hit_ratio": st["hits"] / st["total_edges"] if st["total_edges"] > 0 else 0.0,
        "metrics_avg_tree_nodes": float(np.mean(st["nodes"])),
        "metrics_avg_tree_edges": float(np.mean(st["edges"])),
        "metrics_avg_steiner_ratio": st["steiner"] / sum(st["nodes"]) if sum(st["nodes"]) > 0 else 0,
        "metrics_avg_cost_saving": float(np.mean(st["saving"]))
    }

    print("  -> Validações geradas:")
    print(json.dumps(report, indent=2))
    
    if args.output_dir:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "validate.jsonl").write_text(json.dumps(report) + "\n")

if __name__ == "__main__":
    main()