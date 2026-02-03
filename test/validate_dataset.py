import argparse
import csv
import json
import random
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
            if idx >= len(row):
                raise ValueError("nodes.csv colunas insuficientes")
            v = row[idx].strip()
            if v == "":
                raise ValueError("node_id vazio")
            try:
                ids.append(int(v))
            except Exception:
                raise ValueError("node_id nao-inteiro")
        if not ids:
            raise ValueError("nodes.csv sem dados")
        if len(set(ids)) != len(ids):
            raise ValueError("node_id duplicado")
        return ids, set(ids)

def _iter_graph_pt_files(root: Path):
    return sorted(list(root.rglob("*.pt")))

def _torch_load(path: Path):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch nao esta instalado no ambiente de validacao") from e
    return torch.load(path, map_location="cpu")

def _get_edge_index(obj):
    if hasattr(obj, "edge_index"):
        return obj.edge_index
    if isinstance(obj, dict) and "edge_index" in obj:
        return obj["edge_index"]
    return None

def _get_num_nodes(obj, edge_index):
    if hasattr(obj, "num_nodes") and obj.num_nodes is not None:
        try:
            return int(obj.num_nodes)
        except Exception:
            pass
    if hasattr(obj, "x") and obj.x is not None:
        try:
            return int(obj.x.size(0))
        except Exception:
            pass
    try:
        import torch
        if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
            return int(edge_index.max().item()) + 1
    except Exception:
        pass
    return None

def _validate_graph_obj(obj, nodes_count):
    import torch
    edge_index = _get_edge_index(obj)
    if edge_index is None:
        raise ValueError("edge_index ausente")
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError("edge_index nao e tensor")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index invalido")
    if edge_index.numel() == 0:
        raise ValueError("edge_index vazio")
    if edge_index.dtype not in (torch.int64, torch.int32):
        raise ValueError("edge_index dtype invalido")
    if edge_index.min().item() < 0:
        raise ValueError("edge_index negativo")
    num_nodes = _get_num_nodes(obj, edge_index)
    if num_nodes is None or num_nodes <= 0 or num_nodes > nodes_count:
        raise ValueError("num_nodes invalido")
    if edge_index.max().item() >= num_nodes:
        raise ValueError("edge_index fora de range")
    if hasattr(obj, "x") and obj.x is not None and isinstance(obj.x, torch.Tensor):
        if obj.x.dim() != 2:
            raise ValueError("x invalido")
        if int(obj.x.size(0)) != int(num_nodes):
            raise ValueError("x.size(0) != num_nodes")
        if torch.isnan(obj.x).any().item() or torch.isinf(obj.x).any().item():
            raise ValueError("x possui NaN/Inf")
    if hasattr(obj, "edge_attr") and obj.edge_attr is not None and isinstance(obj.edge_attr, torch.Tensor):
        if obj.edge_attr.dim() != 2:
            raise ValueError("edge_attr invalido")
        if int(obj.edge_attr.size(0)) != int(edge_index.size(1)):
            raise ValueError("edge_attr nao bate com E")
        if torch.isnan(obj.edge_attr).any().item() or torch.isinf(obj.edge_attr).any().item():
            raise ValueError("edge_attr possui NaN/Inf")
    return num_nodes, int(edge_index.size(1)), edge_index

def _parse_snapshots_txt(snap_path: Path):
    refs = []
    base = snap_path.parent
    for ln in snap_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s:
            continue
        direct = base / f"{s}.pt"
        if direct.exists():
            refs.append(direct.resolve())
            continue
        matches = list(base.glob(f"*{s}*.pt"))
        if matches:
            refs.append(matches[0].resolve())
    return refs

def _read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                raise ValueError(f"jsonl invalido em {path} linha {i}")
            if not isinstance(obj, dict):
                raise ValueError(f"jsonl linha {i} nao e objeto")
            items.append(obj)
    if not items:
        raise ValueError(f"jsonl vazio: {path}")
    return items

def _normalize_edge(a: int, b: int):
    return (a, b) if a <= b else (b, a)

def _tqdm(iterable, total=None, desc=None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable

def _build_physical_edge_set(edge_index):
    import torch
    a = edge_index[0].to(torch.int64).tolist()
    b = edge_index[1].to(torch.int64).tolist()
    s = set()
    for i in range(len(a)):
        x = a[i]
        y = b[i]
        if x <= y:
            s.add((x, y))
        else:
            s.add((y, x))
    return s

def _labels_probe_dir(input_dir: Path) -> Path | None:
    direct = input_dir / "labels"
    if direct.exists() and direct.is_dir():
        return direct
    matches = [p for p in input_dir.rglob("labels") if p.is_dir()]
    if matches:
        return matches[0]
    return None

def _extract_terminals_from_instance(obj: dict):
    for k in ["terminals", "terminal_nodes", "terminal", "T", "query_nodes"]:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    return None

def _validate_labels(labels_dir: Path, instances: list[dict], nodes_count: int, physical_edge_set, seed: int = 123):
    jsonl_files = sorted(list(labels_dir.rglob("*.jsonl")))
    if not jsonl_files:
        return None

    labels = _read_jsonl(jsonl_files[0])
    if len(labels) != len(instances):
        raise ValueError("labels nao alinhados com instancias")

    inst_by_id = {}
    inst_missing_id = 0
    for i, inst in enumerate(instances):
        if "id" in inst:
            try:
                inst_id = int(inst["id"])
            except Exception:
                raise ValueError("instances.jsonl possui id nao-inteiro")
        else:
            inst_missing_id += 1
            inst_id = i
        snap = inst.get("snapshot")
        root = inst.get("root")
        terms = _extract_terminals_from_instance(inst)
        if snap is None:
            raise ValueError(f"instancia sem snapshot: id={inst_id}")
        if root is None:
            raise ValueError(f"instancia sem root: id={inst_id}")
        if terms is None:
            raise ValueError(f"instancia sem terminals: id={inst_id}")
        try:
            r = int(root)
        except Exception:
            raise ValueError(f"instancia root nao-inteiro: id={inst_id}")
        t = []
        for v in terms:
            if isinstance(v, bool):
                raise ValueError(f"instancia terminals boolean: id={inst_id}")
            try:
                t.append(int(v))
            except Exception:
                raise ValueError(f"instancia terminal nao-inteiro: id={inst_id}")
        if not t:
            raise ValueError(f"instancia terminals vazio: id={inst_id}")
        if len(set(t)) != len(t):
            raise ValueError(f"instancia terminals duplicados: id={inst_id}")
        inst_by_id[inst_id] = {
            "snapshot": str(snap),
            "root": r,
            "terminals": t,
        }

    checked = 0
    ok_range = 0
    ok_edges_nodes = 0
    ok_terms_covered = 0
    ok_root_covered = 0
    ok_tree = 0
    ok_conn = 0
    ok_unique = 0

    ok_pair_id = 0
    ok_pair_snapshot = 0
    ok_pair_root = 0
    ok_pair_terminals_exact = 0
    ok_pair_terminals_in_tree_nodes = 0
    ok_pair_all_terminals_reachable = 0

    phys_checked = 0
    phys_edge_hits = 0
    phys_edge_total = 0

    it = _tqdm(labels, total=len(labels), desc="validate_labels")
    for i, lab in enumerate(it):
        checked += 1

        if "id" in lab:
            try:
                lab_id = int(lab["id"])
            except Exception:
                raise ValueError("labels.jsonl possui id nao-inteiro")
        else:
            lab_id = i

        inst = inst_by_id.get(lab_id)
        if inst is None:
            raise ValueError(f"label sem instancia correspondente: id={lab_id}")
        ok_pair_id += 1

        snap_lab = lab.get("snapshot")
        if snap_lab is None:
            raise ValueError(f"label sem snapshot: id={lab_id}")
        if str(snap_lab) == inst["snapshot"]:
            ok_pair_snapshot += 1
        else:
            raise ValueError(f"snapshot mismatch: id={lab_id}")

        root_lab = lab.get("root")
        if root_lab is None:
            raise ValueError(f"label sem root: id={lab_id}")
        try:
            r = int(root_lab)
        except Exception:
            raise ValueError(f"label root nao-inteiro: id={lab_id}")
        if r == inst["root"]:
            ok_pair_root += 1
        else:
            raise ValueError(f"root mismatch: id={lab_id}")

        terminals_lab = lab.get("terminals")
        if not isinstance(terminals_lab, list):
            raise ValueError(f"label sem terminals: id={lab_id}")
        t_lab = []
        for v in terminals_lab:
            if isinstance(v, bool):
                raise ValueError(f"label terminals boolean: id={lab_id}")
            try:
                t_lab.append(int(v))
            except Exception:
                raise ValueError(f"label terminal nao-inteiro: id={lab_id}")
        if t_lab == inst["terminals"]:
            ok_pair_terminals_exact += 1
        else:
            if set(t_lab) == set(inst["terminals"]) and len(t_lab) == len(inst["terminals"]):
                ok_pair_terminals_exact += 1
            else:
                raise ValueError(f"terminals mismatch: id={lab_id}")

        tree_nodes = lab.get("tree_nodes")
        tree_edges = lab.get("tree_edges")
        if not isinstance(tree_nodes, list) or not isinstance(tree_edges, list):
            raise ValueError(f"label sem tree_nodes/tree_edges: id={lab_id}")

        t_nodes = []
        for v in tree_nodes:
            if isinstance(v, bool):
                raise ValueError(f"tree_nodes boolean: id={lab_id}")
            try:
                t_nodes.append(int(v))
            except Exception:
                raise ValueError(f"tree_nodes nao-inteiro: id={lab_id}")
        node_set = set(t_nodes)

        ok = True
        if r < 0 or r >= nodes_count:
            ok = False
        if ok:
            for v in node_set:
                if v < 0 or v >= nodes_count:
                    ok = False
                    break
        if ok:
            for v in t_lab:
                if v < 0 or v >= nodes_count:
                    ok = False
                    break
        if ok:
            for e in tree_edges:
                if not (isinstance(e, (list, tuple)) and len(e) == 2):
                    ok = False
                    break
                a = int(e[0])
                b = int(e[1])
                if a < 0 or a >= nodes_count or b < 0 or b >= nodes_count:
                    ok = False
                    break
        if ok:
            ok_range += 1
        else:
            raise ValueError(f"label fora de range: id={lab_id}")

        ok2 = True
        norm_edges = []
        for e in tree_edges:
            a = int(e[0])
            b = int(e[1])
            if a not in node_set or b not in node_set:
                ok2 = False
                break
            norm_edges.append(_normalize_edge(a, b))
        if ok2:
            ok_edges_nodes += 1
        else:
            raise ValueError(f"tree_edges fora de tree_nodes: id={lab_id}")

        if all(v in node_set for v in t_lab):
            ok_terms_covered += 1
        else:
            raise ValueError(f"terminals nao cobertos por tree_nodes: id={lab_id}")

        if r in node_set:
            ok_root_covered += 1
        else:
            raise ValueError(f"root nao coberto por tree_nodes: id={lab_id}")

        if len(set(norm_edges)) == len(norm_edges):
            ok_unique += 1
        else:
            raise ValueError(f"tree_edges duplicados: id={lab_id}")

        if len(norm_edges) == len(node_set) - 1:
            ok_tree += 1
        else:
            raise ValueError(f"nao e arvore (E != V-1): id={lab_id}")

        adj = {v: [] for v in node_set}
        for a, b in norm_edges:
            adj[a].append(b)
            adj[b].append(a)

        start = r
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nx in adj.get(cur, []):
                if nx not in seen:
                    stack.append(nx)

        if len(seen) == len(node_set):
            ok_conn += 1
        else:
            raise ValueError(f"tree nao conectada: id={lab_id}")

        ok_pair_terminals_in_tree_nodes += 1
        ok_pair_all_terminals_reachable += 1

        if norm_edges:
            phys_checked += 1
            for a, b in norm_edges:
                if (a, b) in physical_edge_set:
                    phys_edge_hits += 1
            phys_edge_total += len(norm_edges)

    def ratio(x):
        return (x / checked) if checked else None

    return {
        "labels_format": "jsonl",
        "labels_files_found": len(labels),
        "labels_instances_aligned": True,
        "labels_schema": "tree_nodes+tree_edges",
        "labels_tree_in_range": ratio(ok_range),
        "labels_tree_edges_in_nodes": ratio(ok_edges_nodes),
        "labels_tree_terminals_covered": ratio(ok_terms_covered),
        "labels_tree_root_covered": ratio(ok_root_covered),
        "labels_tree_edges_unique": ratio(ok_unique),
        "labels_tree_is_tree": ratio(ok_tree),
        "labels_tree_connected": ratio(ok_conn),
        "labels_pair_id_match": ratio(ok_pair_id),
        "labels_pair_snapshot_match": ratio(ok_pair_snapshot),
        "labels_pair_root_match": ratio(ok_pair_root),
        "labels_pair_terminals_match": ratio(ok_pair_terminals_exact),
        "labels_pair_terminals_in_tree_nodes": ratio(ok_pair_terminals_in_tree_nodes),
        "labels_pair_all_terminals_reachable": ratio(ok_pair_all_terminals_reachable),
        "labels_physical_checked_instances": phys_checked,
        "labels_physical_edge_hit_ratio": (phys_edge_hits / phys_edge_total) if phys_edge_total > 0 else None,
    }

def validate_dataset(input_dir: Path):
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise ValueError("input_dir nao existe")

    nodes_path = _find_file(input_dir, "nodes.csv")
    snaps_path = _find_file(input_dir, "snapshots.txt")
    inst_path = _find_file(input_dir, "instances.jsonl")

    if nodes_path is None:
        raise ValueError("nodes.csv nao encontrado")
    if snaps_path is None:
        raise ValueError("snapshots.txt nao encontrado")
    if inst_path is None:
        raise ValueError("instances.jsonl nao encontrado")

    node_ids, node_set = _load_nodes_csv(nodes_path)

    pt_files = _iter_graph_pt_files(input_dir)
    if not pt_files:
        raise ValueError("nenhum .pt encontrado")

    snap_refs = _parse_snapshots_txt(snaps_path)
    if not snap_refs:
        raise ValueError("snapshots.txt nao referencia nenhum .pt")
    for p in snap_refs:
        if not p.exists():
            raise ValueError(f"snapshot inexistente: {p}")

    graphs_checked = 0
    total_nodes = 0
    total_edges = 0
    first_edge_index = None

    it = _tqdm(snap_refs, total=len(snap_refs), desc="validate_snapshots")
    for p in it:
        obj = _torch_load(p)
        n, e, edge_index = _validate_graph_obj(obj, len(node_ids))
        graphs_checked += 1
        total_nodes += n
        total_edges += e
        if first_edge_index is None:
            first_edge_index = edge_index

    instances = _read_jsonl(inst_path)

    min_k = None
    max_k = None
    it2 = _tqdm(instances, total=len(instances), desc="validate_instances")
    for obj in it2:
        terminals = _extract_terminals_from_instance(obj)
        if terminals is None:
            raise ValueError("instancia sem terminals")
        t = []
        for v in terminals:
            if isinstance(v, bool):
                raise ValueError("terminal boolean")
            t.append(int(v))
        if not t:
            raise ValueError("terminals vazio")
        if len(set(t)) != len(t):
            raise ValueError("terminals duplicados")
        for v in t:
            if v not in node_set:
                raise ValueError("terminal fora de nodes.csv")
        kk = len(t)
        min_k = kk if min_k is None else min(min_k, kk)
        max_k = kk if max_k is None else max(max_k, kk)

    if first_edge_index is None:
        raise ValueError("edge_index nao encontrado")
    physical_edge_set = _build_physical_edge_set(first_edge_index)

    labels_dir = _labels_probe_dir(input_dir)
    labels_stats = None
    if labels_dir is not None:
        labels_stats = _validate_labels(
            labels_dir=labels_dir,
            instances=instances,
            nodes_count=len(node_ids),
            physical_edge_set=physical_edge_set,
            seed=123,
        )

    report = {
        "input_dir": str(input_dir),
        "nodes_count": len(node_ids),
        "graphs_found": len(pt_files),
        "graphs_checked": graphs_checked,
        "avg_nodes_per_graph_checked": (total_nodes / graphs_checked) if graphs_checked else None,
        "avg_edges_per_graph_checked": (total_edges / graphs_checked) if graphs_checked else None,
        "instances_found": len(instances),
        "instances_checked": len(instances),
        "instances_with_terminals": len(instances),
        "instances_terminals_k_min": min_k,
        "instances_terminals_k_max": max_k,
        "snapshot_pt_refs_found": len(snap_refs),
        "check_physical_edges": True,
        "physical_edge_set_size": len(physical_edge_set),
    }

    if labels_stats is not None:
        report.update(labels_stats)

    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    args = ap.parse_args()

    report = validate_dataset(Path(args.input_dir))

    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)