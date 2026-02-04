import argparse
import json
import random
import torch
from pathlib import Path
from collections import deque
from tqdm import tqdm

NUM_INTERESTS = 5
K_MIN = 10
K_MAX = 30
RADIUS_HOPS = 3
KEEP_RATIO = 0.7
SEED = 42

def list_graphs(input_dir):
    return sorted(Path(input_dir).glob("as_graph_*.pt"))

def build_adj_undirected(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u, v = int(edge_index[0, i].item()), int(edge_index[1, i].item())
        adj[u].add(v); adj[v].add(u)
    return adj

def reachable_from_root(adj, root):
    root = int(root)
    if root < 0 or root >= len(adj): return set()
    seen, q = set(), deque([root])
    while q:
        u = q.popleft()
        if u not in seen:
            seen.add(u)
            for v in adj[u]:
                if v not in seen: q.append(v)
    return seen

def nodes_within_hops(adj, sources, max_hops):
    seen, q = set(sources), deque([(s, 0) for s in sources])
    while q:
        u, d = q.popleft()
        if d < max_hops:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append((v, d + 1))
    return seen

def sample_local(antennas_set, candidates_set, k, rng):
    candidates = list(candidates_set & antennas_set)
    if len(candidates) >= k: return set(rng.sample(candidates, k))
    out = set(candidates)
    rem = k - len(out)
    pool = list(antennas_set - out)
    if pool and rem > 0: out |= set(rng.sample(pool, k=min(rem, len(pool))))
    return out

def evolve_active(adj, antennas_set, active_prev, k_next, rng):
    prev = list(active_prev & antennas_set)
    if not prev:
        seed = rng.choice(list(antennas_set))
        cand = nodes_within_hops(adj, [seed], RADIUS_HOPS)
        return sample_local(antennas_set, cand, k_next, rng)

    keep_k = max(1, min(int(round(len(prev) * KEEP_RATIO)), len(prev)))
    keep = set(rng.sample(prev, keep_k))
    cand = (nodes_within_hops(adj, list(keep), RADIUS_HOPS) & antennas_set) - keep
    
    needed = k_next - len(keep)
    add = set()
    if needed > 0:
        c_list = list(cand)
        if c_list: add |= set(rng.sample(c_list, k=min(needed, len(c_list))))
    
    needed2 = k_next - (len(keep) + len(add))
    if needed2 > 0:
        pool = list(antennas_set - keep - add)
        if pool: add |= set(rng.sample(pool, k=min(needed2, len(pool))))
            
    return keep | add

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True); ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    graphs = list_graphs(args.input_dir)
    if not graphs: return
    rng = random.Random(SEED)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup
    g0 = torch.load(graphs[0], map_location="cpu")
    antennas_all = [n for n, t in g0["node_types"].items() if t == "antena"]
    snaps, roots = [], []
    for gp in graphs:
        g = torch.load(gp, map_location="cpu")
        snaps.append(g["snapshot"]); roots.append(g["server_id"])

    rec_id = 0
    active_by_interest = {}
    steps = max(1, len(snaps) - 1)
    
    with open(out_dir / "instances.jsonl", "w", encoding="utf-8") as f:
        with tqdm(total=steps*NUM_INTERESTS, desc="generate_instances: escrevendo instances.jsonl", unit="instância") as pbar:
            for t in range(steps):
                idx_curr, idx_next = (0, 0) if len(snaps) < 2 else (t, t+1)
                g_next = torch.load(graphs[idx_next], map_location="cpu")
                
                if t >= 2: history = snaps[t-2:t+1]
                elif t == 1: history = snaps[0:2]
                else: history = [snaps[0]]

                adj = build_adj_undirected(g_next["edge_index"], g_next["num_nodes"])
                reach = reachable_from_root(adj, roots[idx_next])
                antennas_set = (set(antennas_all) & reach) or set(antennas_all)

                for iid in range(NUM_INTERESTS):
                    if iid not in active_by_interest:
                        k0 = rng.randint(K_MIN, min(K_MAX, len(antennas_set)))
                        seed = rng.choice(list(antennas_set))
                        active_by_interest[iid] = sample_local(antennas_set, nodes_within_hops(adj, [seed], RADIUS_HOPS), k0, rng)
                    
                    active_by_interest[iid] &= antennas_set
                    if not active_by_interest[iid]:
                         k0 = rng.randint(K_MIN, min(K_MAX, len(antennas_set)))
                         active_by_interest[iid] = sample_local(antennas_set, nodes_within_hops(adj, [rng.choice(list(antennas_set))], RADIUS_HOPS), k0, rng)

                    k_next = max(K_MIN, min(K_MAX, len(active_by_interest[iid]) + rng.randint(-3, 3), len(antennas_set)))
                    terminals_out = sorted(list(evolve_active(adj, antennas_set, active_by_interest[iid], k_next, rng)))
                    active_by_interest[iid] = set(terminals_out)

                    f.write(json.dumps({
                        "id": rec_id, "snapshot": snaps[idx_curr], "snapshot_next": snaps[idx_next],
                        "history_snapshots": history, "root": roots[idx_next], "interest_id": iid,
                        "radius_hops": RADIUS_HOPS, "terminals_in": sorted(list(active_by_interest[iid])), "terminals_out": terminals_out
                    }) + "\n")
                    rec_id += 1; pbar.update(1)
    print(f"  -> Instances geradas: {rec_id}\n")

if __name__ == "__main__": main()