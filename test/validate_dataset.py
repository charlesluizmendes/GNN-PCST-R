import argparse
import csv
import json
import sys
from pathlib import Path

def _find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    return None

def _load_nodes_csv(nodes_path: Path):
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("nodes.csv vazio")
        header_norm = [h.strip().lower() for h in header]
        candidates = ["node_id", "id", "node", "asn", "as", "index"]
        idx = None
        for c in candidates:
            if c in header_norm:
                idx = header_norm.index(c)
                break
        if idx is None:
            idx = 0
        ids = []
        for row in reader:
            if not row:
                continue
            v = row[idx].strip()
            if v == "":
                raise ValueError("node_id vazio")
            ids.append(int(v))
        if not ids:
            raise ValueError("nodes.csv sem dados")
        if len(set(ids)) != len(ids):
            raise ValueError("node_id duplicado")
        return ids, set(ids)

def _iter_graph_pt_files(root: Path):
    return sorted(list(root.rglob("*.pt")))

def _torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu")

def _get_edge_index(obj):
    if hasattr(obj, "edge_index"):
        return obj.edge_index
    if isinstance(obj, dict) and "edge_index" in obj:
        return obj["edge_index"]
    return None

def _get_num_nodes(obj, edge_index):
    if hasattr(obj, "num_nodes") and obj.num_nodes is not None:
        return int(obj.num_nodes)
    if hasattr(obj, "x") and obj.x is not None:
        return int(obj.x.size(0))
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return None

def _validate_graph_obj(obj, nodes_count):
    import torch
    edge_index = _get_edge_index(obj)
    if edge_index is None:
        raise ValueError("edge_index ausente")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index invalido")
    if edge_index.min().item() < 0:
        raise ValueError("edge_index negativo")
    num_nodes = _get_num_nodes(obj, edge_index)
    if num_nodes is None or num_nodes > nodes_count:
        raise ValueError("num_nodes invalido")
    if edge_index.max().item() >= num_nodes:
        raise ValueError("edge_index fora de range")
    if hasattr(obj, "x") and obj.x is not None:
        if torch.isnan(obj.x).any().item() or torch.isinf(obj.x).any().item():
            raise ValueError("x invalido")
    if hasattr(obj, "edge_attr") and obj.edge_attr is not None:
        if torch.isnan(obj.edge_attr).any().item() or torch.isinf(obj.edge_attr).any().item():
            raise ValueError("edge_attr invalido")
    return num_nodes, int(edge_index.size(1))

def _parse_snapshots_txt(snap_path: Path, root: Path):
    refs = []
    base = snap_path.parent
    for ln in snap_path.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        p = base / f"{s}.pt"
        if p.exists():
            refs.append(p.resolve())
            continue
        matches = list(base.glob(f"*{s}*.pt"))
        if matches:
            refs.append(matches[0].resolve())
    return refs

def _read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError("linha json invalida")
            items.append(obj)
    if not items:
        raise ValueError("jsonl vazio")
    return items

def _normalize_edge(a, b):
    return (a, b) if a <= b else (b, a)

def _validate_labels(labels_dir: Path, instances_count: int, nodes_count: int):
    jsonl_files = list(labels_dir.rglob("*.jsonl"))
    if not jsonl_files:
        return None
    items = _read_jsonl(jsonl_files[0])
    if len(items) != instances_count:
        raise ValueError("labels nao alinhados com instancias")
    checked = 0
    ok_range = ok_edges_nodes = ok_terms = ok_root = ok_tree = ok_conn = ok_unique = 0
    for obj in items[:2000]:
        checked += 1
        root = int(obj["root"])
        terminals = [int(x) for x in obj["terminals"]]
        tree_nodes = [int(x) for x in obj["tree_nodes"]]
        tree_edges = [(int(a), int(b)) for a, b in obj["tree_edges"]]
        node_set = set(tree_nodes)
        if root in node_set and all(0 <= v < nodes_count for v in node_set):
            ok_range += 1
        if all(a in node_set and b in node_set for a, b in tree_edges):
            ok_edges_nodes += 1
        if all(t in node_set for t in terminals):
            ok_terms += 1
        if root in node_set:
            ok_root += 1
        norm_edges = [_normalize_edge(a, b) for a, b in tree_edges]
        if len(set(norm_edges)) == len(norm_edges):
            ok_unique += 1
        if len(norm_edges) == len(node_set) - 1:
            ok_tree += 1
        adj = {v: [] for v in node_set}
        for a, b in norm_edges:
            adj[a].append(b)
            adj[b].append(a)
        seen = set()
        stack = [root]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            for n in adj.get(v, []):
                if n not in seen:
                    stack.append(n)
        if len(seen) == len(node_set):
            ok_conn += 1
    def r(x):
        return x / checked if checked else None
    return {
        "labels_format": "jsonl",
        "labels_files_found": len(items),
        "labels_instances_aligned": True,
        "labels_schema": "tree_nodes+tree_edges",
        "labels_tree_in_range": r(ok_range),
        "labels_tree_edges_in_nodes": r(ok_edges_nodes),
        "labels_tree_terminals_covered": r(ok_terms),
        "labels_tree_root_covered": r(ok_root),
        "labels_tree_edges_unique": r(ok_unique),
        "labels_tree_is_tree": r(ok_tree),
        "labels_tree_connected": r(ok_conn),
    }

def validate_dataset(input_dir: Path):
    input_dir = input_dir.resolve()
    nodes_path = _find_file(input_dir, "nodes.csv")
    snaps_path = _find_file(input_dir, "snapshots.txt")
    inst_path = _find_file(input_dir, "instances.jsonl")
    node_ids, node_set = _load_nodes_csv(nodes_path)
    pt_files = _iter_graph_pt_files(input_dir)
    snap_refs = _parse_snapshots_txt(snaps_path, input_dir)
    graphs_checked = 0
    total_nodes = 0
    total_edges = 0
    for p in snap_refs:
        obj = _torch_load(p)
        n, e = _validate_graph_obj(obj, len(node_ids))
        graphs_checked += 1
        total_nodes += n
        total_edges += e
    instances = _read_jsonl(inst_path)
    labels_dir = input_dir / "labels"
    labels_stats = _validate_labels(labels_dir, len(instances), len(node_ids)) if labels_dir.exists() else None
    report = {
        "input_dir": str(input_dir),
        "nodes_count": len(node_ids),
        "graphs_found": len(pt_files),
        "graphs_checked": graphs_checked,
        "avg_nodes_per_graph_checked": total_nodes / graphs_checked,
        "avg_edges_per_graph_checked": total_edges / graphs_checked,
        "instances_found": len(instances),
        "instances_checked": len(instances),
        "instances_with_terminals": len(instances),
        "snapshot_pt_refs_found": len(snap_refs),
    }
    if labels_stats:
        report.update(labels_stats)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    report = validate_dataset(Path(args.input_dir))

    out_pretty = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validate.jsonl").write_text(out_pretty + "\n", encoding="utf-8")

    print(out_pretty)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)
