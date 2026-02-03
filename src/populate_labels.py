import argparse
import json
from pathlib import Path
import torch
from pcst_fast import pcst_fast as pcst

def call_pcst(fn, edges, prizes, costs, root):
    trials = [
        (edges, prizes, costs, root),
        (edges, prizes, costs, root, 1),
        (edges, prizes, costs, root, 1, "gw"),
        (edges, prizes, costs, root, 1, "strong"),
        (edges, prizes, costs, root, 1, "gw", 0),
        (edges, prizes, costs, root, 1, "strong", 0),
        (edges, prizes, costs, root, 1, 0),
        (edges, prizes, costs, root, 1, 0, 0),
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

def build_undirected_edges(g, cost_p2p, cost_p2c):
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
            costs.append(float(cost_p2p))
        else:
            costs.append(float(cost_p2c))

    edges_t = torch.tensor(edges, dtype=torch.int64)
    costs_t = torch.tensor(costs, dtype=torch.float32)
    return edges_t, costs_t

def iter_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, nargs=2)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_instances", type=int, default=-1)
    ap.add_argument("--cost_p2p", type=float, default=1.0)
    ap.add_argument("--cost_p2c", type=float, default=1.2)
    ap.add_argument("--terminal_prize", type=float, default=1000000.0)
    ap.add_argument("--root_prize", type=float, default=1000000.0)
    args = ap.parse_args()

    snapshots_dir = Path(args.input_dir[0])
    instances_dir = Path(args.input_dir[1])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances_path = instances_dir / "instances.jsonl"
    if not instances_path.exists():
        raise FileNotFoundError(str(instances_path))

    graph_cache = {}
    edge_cache = {}

    out_path = out_dir / "labels.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for inst in iter_instances(instances_path):
            if args.max_instances > 0 and written >= args.max_instances:
                break

            snap = inst["snapshot"]
            gid = f"as_graph_{snap}.pt"
            gp = snapshots_dir / gid

            if snap not in graph_cache:
                if not gp.exists():
                    raise FileNotFoundError(str(gp))
                g = load_graph(gp)
                graph_cache[snap] = g
            else:
                g = graph_cache[snap]

            n = int(g["num_nodes"])

            if snap not in edge_cache:
                edges_u, costs_u = build_undirected_edges(g, args.cost_p2p, args.cost_p2c)
                edge_cache[snap] = (edges_u, costs_u)
            else:
                edges_u, costs_u = edge_cache[snap]

            prizes = torch.zeros(n, dtype=torch.float32)
            root = int(inst["root"])
            terminals = [int(x) for x in inst["terminals"]]

            touched = []
            prizes[root] = float(args.root_prize)
            touched.append(root)
            for t in terminals:
                prizes[t] = float(args.terminal_prize)
                touched.append(t)

            sel_nodes, sel_edges = call_pcst(pcst, edges_u, prizes, costs_u, root)

            prizes[touched] = 0.0

            if isinstance(sel_nodes, torch.Tensor):
                sel_nodes = sel_nodes.tolist()
            if isinstance(sel_edges, torch.Tensor):
                sel_edges = sel_edges.tolist()

            tree_edges = []
            for ei_idx in sel_edges:
                a = int(edges_u[int(ei_idx)][0].item())
                b = int(edges_u[int(ei_idx)][1].item())
                tree_edges.append([a, b])

            rec = {
                "id": int(inst["id"]),
                "snapshot": snap,
                "root": root,
                "terminals": terminals,
                "tree_nodes": [int(x) for x in sel_nodes],
                "tree_edges": tree_edges
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("DONE", written, str(out_path))

if __name__ == "__main__":
    main()
