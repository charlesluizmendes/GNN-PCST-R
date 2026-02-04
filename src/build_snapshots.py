import argparse
import bz2
import csv
import statistics
import torch
import re
from pathlib import Path
from tqdm import tqdm

def open_text(path):
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")

def iter_records(path):
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("|")
            if len(parts) < 3: continue
            yield int(parts[0]), int(parts[1]), int(parts[2])

def snapshot_name(path):
    name = Path(path).name
    return name.split(".")[0]

def build_adjacency_list(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]
    for i in tqdm(range(edge_index.size(1)), desc="build_snapshots: construindo grafo", unit="aresta"):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        adj[u].add(v); adj[v].add(u)
    return adj

def identify_server(adj, num_nodes):
    degrees = [len(adj[node]) for node in range(num_nodes)]
    server_id = degrees.index(max(degrees))
    print(f"  -> Servidor: node_id={server_id}, grau={degrees[server_id]}")
    return server_id, degrees

def identify_peripheral_nodes_adaptive(adj, num_nodes, degrees):
    valid_degrees = [d for d in degrees if d > 0]
    if len(valid_degrees) >= 4:
        q1 = statistics.quantiles(valid_degrees, n=4)[0]
    elif len(valid_degrees) > 0:
        q1 = float(min(valid_degrees))
    else:
        q1 = 0.0

    neighbor_degree_by_node = []
    for node in range(num_nodes):
        if degrees[node] > 0:
            neighbor_degrees = [degrees[n] for n in adj[node]]
            if neighbor_degrees:
                neighbor_degree_by_node.append(sum(neighbor_degrees) / len(neighbor_degrees))
    
    median_neighbor_degree = statistics.median(neighbor_degree_by_node) if neighbor_degree_by_node else 0
    
    peripheral = set()
    for node in tqdm(range(num_nodes), desc="build_snapshots: classificando antenas", unit="nó"):
        degree = degrees[node]
        if degree == 0: continue
        if degree == 1:
            peripheral.add(node)
        elif degree <= q1:
            vals = [degrees[n] for n in adj[node]]
            if vals and (sum(vals) / len(vals)) > median_neighbor_degree:
                peripheral.add(node)
    
    print(f"  -> Antenas identificadas: {len(peripheral)} nós")
    return peripheral

def identify_node_types(edge_index, num_nodes):
    adj = build_adjacency_list(edge_index, num_nodes)
    server_id, degrees = identify_server(adj, num_nodes)
    peripheral = identify_peripheral_nodes_adaptive(adj, num_nodes, degrees)
    
    node_types = {}
    for node in range(num_nodes):
        if node == server_id: node_types[node] = "servidor"
        elif node in peripheral: node_types[node] = "antena"
        else: node_types[node] = "roteador"
    return node_types, server_id

def build_edge_index_from_file(path, asn2id):
    src, dst, et = [], [], []
    for a, b, rel in iter_records(path):
        ia, ib = asn2id[a], asn2id[b]
        if rel == 0:
            src.extend([ia, ib]); dst.extend([ib, ia]); et.extend([0, 0])
        elif rel == -1:
            src.extend([ia, ib]); dst.extend([ib, ia]); et.extend([1, 2])
            
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(et, dtype=torch.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.txt")) + list(input_dir.glob("*.txt.bz2")))
    if not files: return

    nodes = set()
    for p in files:
        for a, b, _ in iter_records(p): nodes.add(a); nodes.add(b)
    asn2id = {asn: i for i, asn in enumerate(sorted(nodes))}

    with open(output_dir / "snapshots.txt", "w", encoding="utf-8") as f:
        for p in tqdm(files, desc="build_snapshots: salvando snapshots.txt", unit="snapshot"):
            f.write(snapshot_name(p) + "\n")

    edge_index_first, _ = build_edge_index_from_file(files[0], asn2id)
    node_types, server_id = identify_node_types(edge_index_first, len(asn2id))

    last_snapshot_name = None
    for p in tqdm(files, desc="build_snapshots: gerando grafos .pt", unit="snapshot"):
        edge_index, edge_type = build_edge_index_from_file(p, asn2id)
        current_name = snapshot_name(p)
        ts = int(re.search(r'\d+', current_name).group()) if re.search(r'\d+', current_name) else 0

        obj = {
            "snapshot": current_name,
            "timestamp": ts,
            "prev_snapshot": last_snapshot_name,
            "num_nodes": len(asn2id),
            "edge_index": edge_index,
            "edge_type": edge_type,
            "server_id": server_id,
            "node_types": node_types
        }
        torch.save(obj, output_dir / f"as_graph_{current_name}.pt")
        last_snapshot_name = current_name

    with open(output_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "asn", "tipo"])
        for asn, node_id in tqdm(asn2id.items(), desc="build_snapshots: salvando nodes.csv", unit="nó"):
            w.writerow([node_id, asn, node_types.get(node_id, "roteador")])
    print()

if __name__ == "__main__": main()