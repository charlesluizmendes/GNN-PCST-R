import argparse
import bz2
import csv
import statistics
import torch
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
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            a = int(parts[0])
            b = int(parts[1])
            rel = int(parts[2])
            yield a, b, rel

def snapshot_name(path):
    name = Path(path).name
    return name.split(".")[0]

def build_adjacency_list(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]
    for i in tqdm(range(edge_index.size(1)), desc="build_snapshots: construindo grafo", unit="aresta"):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        adj[u].add(v)
    return adj

def identify_server(adj, num_nodes):
    degrees = []
    for node in tqdm(range(num_nodes), desc="build_snapshots: calculando graus", unit="nó"):
        degrees.append(len(adj[node]))
    
    server_id = degrees.index(max(degrees))
    max_degree = degrees[server_id]
    
    print(f"  -> Servidor: node_id={server_id}, grau={max_degree}")
    
    return server_id, degrees

def identify_peripheral_nodes_adaptive(adj, num_nodes, degrees):
    valid_degrees = [d for d in degrees if d > 0]
    
    q1 = statistics.quantiles(valid_degrees, n=4)[0]
    q2 = statistics.quantiles(valid_degrees, n=4)[1]
    q3 = statistics.quantiles(valid_degrees, n=4)[2]
    
    print(f"  -> Distribuição de graus: Q1={q1:.1f}, Mediana={q2:.1f}, Q3={q3:.1f}")
    
    neighbor_degree_by_node = []
    for node in tqdm(range(num_nodes), desc="build_snapshots: analisando vizinhanças", unit="nó"):
        if degrees[node] > 0:
            neighbor_degrees = [degrees[n] for n in adj[node]]
            avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)
            neighbor_degree_by_node.append(avg_neighbor_degree)
    
    if neighbor_degree_by_node:
        median_neighbor_degree = statistics.median(neighbor_degree_by_node)
    else:
        median_neighbor_degree = 0
    
    print(f"  -> Mediana do grau médio dos vizinhos: {median_neighbor_degree:.1f}")
    
    peripheral = set()
    
    for node in tqdm(range(num_nodes), desc="build_snapshots: classificando antenas", unit="nó"):
        degree = degrees[node]
        
        if degree == 0:
            continue
        
        if degree == 1:
            peripheral.add(node)
            continue
        
        if degree <= q1:
            neighbor_degrees = [degrees[n] for n in adj[node]]
            avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)
            
            if avg_neighbor_degree > median_neighbor_degree:
                peripheral.add(node)
    
    print(f"  -> Antenas identificadas: {len(peripheral)} nós")
    
    return peripheral

def identify_node_types(edge_index, num_nodes):
    adj = build_adjacency_list(edge_index, num_nodes)
    
    server_id, degrees = identify_server(adj, num_nodes)
    
    peripheral = identify_peripheral_nodes_adaptive(adj, num_nodes, degrees)
    
    node_types = {}
    for node in range(num_nodes):
        if node == server_id:
            node_types[node] = "servidor"
        elif node in peripheral:
            node_types[node] = "antena"
        else:
            node_types[node] = "roteador"
    
    return node_types, server_id

def build_edge_index_from_file(path, asn2id):
    src = []
    dst = []
    et = []

    for a, b, rel in iter_records(path):
        ia = asn2id[a]
        ib = asn2id[b]

        if rel == 0:
            src.append(ia); dst.append(ib); et.append(0)
            src.append(ib); dst.append(ia); et.append(0)
        elif rel == -1:
            src.append(ia); dst.append(ib); et.append(1)
            src.append(ib); dst.append(ia); et.append(2)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(et, dtype=torch.uint8)
    
    return edge_index, edge_type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(input_dir.glob("*.txt")) + list(input_dir.glob("*.txt.bz2")))
    if not files:
        return

    nodes = set()
    for p in files:
        for a, b, rel in iter_records(p):
            nodes.add(a)
            nodes.add(b)

    asn_list = sorted(nodes)
    asn2id = {asn: i for i, asn in enumerate(asn_list)}

    snaps_path = output_dir / "snapshots.txt"
    with open(snaps_path, "w", encoding="utf-8") as f:
        for p in tqdm(files, desc="build_snapshots: salvando snapshots.txt", unit="snapshot"):
            f.write(snapshot_name(p) + "\n")

    edge_index_first, _ = build_edge_index_from_file(files[0], asn2id)
    node_types, server_id = identify_node_types(edge_index_first, len(asn2id))

    for p in tqdm(files, desc="build_snapshots: gerando grafos .pt", unit="snapshot"):
        edge_index, edge_type = build_edge_index_from_file(p, asn2id)

        obj = {
            "snapshot": snapshot_name(p),
            "num_nodes": len(asn2id),
            "edge_index": edge_index,
            "edge_type": edge_type,
            "server_id": server_id,
            "node_types": node_types
        }

        out_path = output_dir / f"as_graph_{obj['snapshot']}.pt"
        torch.save(obj, out_path)

    nodes_path = output_dir / "nodes.csv"
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "asn", "tipo"])
        for asn, node_id in tqdm(asn2id.items(), total=len(asn2id), desc="build_snapshots: salvando nodes.csv", unit="nó"):
            tipo = node_types.get(node_id, "roteador")
            w.writerow([node_id, asn, tipo])
    
    print()

if __name__ == "__main__":
    main()