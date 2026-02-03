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
    d = Path(input_dir)
    return sorted(d.glob("as_graph_*.pt"))

def build_adj_undirected(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].add(v)
            adj[v].add(u)
    return adj

def reachable_from_root(adj, root):
    root = int(root)
    if root < 0 or root >= len(adj):
        return set()
    seen = set()
    q = deque([root])
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                q.append(v)
    return seen

def nodes_within_hops(adj, sources, max_hops):
    if max_hops <= 0:
        return set(sources)
    seen = set(sources)
    q = deque([(s, 0) for s in sources])
    while q:
        u, d = q.popleft()
        if d == max_hops:
            continue
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return seen

def sample_local(antennas_set, candidates_set, k, rng):
    candidates = list(candidates_set & antennas_set)
    if len(candidates) >= k:
        return set(rng.sample(candidates, k))
    out = set(candidates)
    remaining = k - len(out)
    pool = list(antennas_set - out)
    if pool and remaining > 0:
        out |= set(rng.sample(pool, k=min(remaining, len(pool))))
    return out

def evolve_active(adj, antennas_set, active_prev, k_next, rng):
    prev = list(active_prev & antennas_set)
    if not prev:
        seed = rng.choice(list(antennas_set))
        cand = nodes_within_hops(adj, [seed], RADIUS_HOPS)
        return sample_local(antennas_set, cand, k_next, rng)

    keep_k = int(round(len(prev) * KEEP_RATIO))
    keep_k = max(1, min(keep_k, len(prev)))
    keep = set(rng.sample(prev, keep_k))

    cand = nodes_within_hops(adj, list(keep), RADIUS_HOPS)
    cand = (cand & antennas_set) - keep

    needed = k_next - len(keep)
    if needed <= 0:
        return set(rng.sample(list(keep), k_next))

    add = set()
    cand_list = list(cand)
    if cand_list:
        add |= set(rng.sample(cand_list, k=min(needed, len(cand_list))))

    needed2 = k_next - (len(keep) + len(add))
    if needed2 > 0:
        pool = list(antennas_set - keep - add)
        if pool:
            add |= set(rng.sample(pool, k=min(needed2, len(pool))))

    return keep | add

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    graphs = list_graphs(args.input_dir)
    if not graphs:
        return

    rng = random.Random(SEED)

    snaps = []
    roots = []
    node_types_ref = None
    for gp in graphs:
        g = torch.load(gp, map_location="cpu")
        snaps.append(str(g.get("snapshot", "")))
        roots.append(int(g.get("server_id", 0)))
        if node_types_ref is None:
            node_types_ref = g.get("node_types", {})

    antennas_all = [int(nid) for nid, t in (node_types_ref or {}).items() if t == "antena"]
    if not antennas_all:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "instances.jsonl"

    active_by_interest = {}
    k_by_interest = {}

    steps = len(snaps) - 1
    if steps <= 0:
        steps = 1

    total = steps * NUM_INTERESTS
    rec_id = 0

    with open(out_path, "w", encoding="utf-8") as f:
        with tqdm(total=total, desc="generate_instances: escrevendo instances.jsonl", unit="instância") as pbar:
            for t in range(steps):
                if len(snaps) >= 2:
                    snap_t = snaps[t]
                    snap_next = snaps[t + 1]
                    root_next = int(roots[t + 1])
                    g_next = torch.load(graphs[t + 1], map_location="cpu")
                else:
                    snap_t = snaps[0]
                    snap_next = snaps[0]
                    root_next = int(roots[0])
                    g_next = torch.load(graphs[0], map_location="cpu")

                num_nodes = int(g_next.get("num_nodes", 0))
                adj = build_adj_undirected(g_next["edge_index"], num_nodes)
                reach = reachable_from_root(adj, root_next)

                antennas_step = [a for a in antennas_all if a in reach]
                if not antennas_step:
                    antennas_step = list(antennas_all)
                antennas_set_step = set(antennas_step)

                for iid in range(NUM_INTERESTS):
                    if iid not in active_by_interest:
                        k0 = rng.randint(K_MIN, min(K_MAX, len(antennas_step)))
                        k_by_interest[iid] = k0
                        seed = rng.choice(antennas_step)
                        cand = nodes_within_hops(adj, [seed], RADIUS_HOPS)
                        active_by_interest[iid] = sample_local(antennas_set_step, cand, k0, rng)

                    active_by_interest[iid] = active_by_interest[iid] & antennas_set_step
                    if not active_by_interest[iid]:
                        k0 = rng.randint(K_MIN, min(K_MAX, len(antennas_step)))
                        k_by_interest[iid] = k0
                        seed = rng.choice(antennas_step)
                        cand = nodes_within_hops(adj, [seed], RADIUS_HOPS)
                        active_by_interest[iid] = sample_local(antennas_set_step, cand, k0, rng)

                    k_prev = int(k_by_interest.get(iid, K_MIN))
                    delta = rng.randint(-3, 3)
                    k_next = max(K_MIN, min(K_MAX, k_prev + delta, len(antennas_step)))

                    terminals_in = sorted(list(active_by_interest[iid]))
                    terminals_out = sorted(list(evolve_active(adj, antennas_set_step, active_by_interest[iid], k_next, rng)))

                    active_by_interest[iid] = set(terminals_out)
                    k_by_interest[iid] = k_next

                    rec = {
                        "id": rec_id,
                        "snapshot": snap_t,
                        "snapshot_next": snap_next,
                        "root": int(root_next),
                        "interest_id": int(iid),
                        "radius_hops": int(RADIUS_HOPS),
                        "terminals_in": terminals_in,
                        "terminals_out": terminals_out
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rec_id += 1
                    pbar.update(1)

    print(f"  -> Instances geradas: {rec_id}")
    print()

if __name__ == "__main__":
    main()
