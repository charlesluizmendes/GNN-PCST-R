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

def _load_nodes_csv(nodes_path: Path) -> tuple[list[str], list[int], set[int]]:
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"nodes.csv vazio: {nodes_path}")
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
                raise ValueError("nodes.csv possui linha com colunas insuficientes")
            v = row[idx].strip()
            if v == "":
                raise ValueError("nodes.csv possui node_id vazio")
            try:
                ids.append(int(v))
            except Exception:
                raise ValueError(f"nodes.csv possui node_id nao-inteiro: {v}")

        if not ids:
            raise ValueError("nodes.csv sem linhas de dados")

        ids_set = set(ids)
        if len(ids_set) != len(ids):
            raise ValueError("nodes.csv possui node_id duplicado")

        return header, ids, ids_set

def _iter_graph_pt_files(root: Path) -> list[Path]:
    pts = list(root.rglob("*.pt"))
    return sorted(pts)

def _torch_load(path: Path):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch nao esta instalado no ambiente de validacao") from e
    return torch.load(path, map_location="cpu")

def _get_edge_index(obj):
    if hasattr(obj, "edge_index"):
        return getattr(obj, "edge_index")
    if isinstance(obj, dict) and "edge_index" in obj:
        return obj["edge_index"]
    return None

def _get_num_nodes(obj, edge_index):
    if hasattr(obj, "num_nodes") and getattr(obj, "num_nodes") is not None:
        try:
            return int(getattr(obj, "num_nodes"))
        except Exception:
            pass
    if hasattr(obj, "x") and getattr(obj, "x") is not None:
        x = getattr(obj, "x")
        try:
            return int(x.size(0))
        except Exception:
            pass
    if edge_index is not None:
        try:
            import torch
            if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
                return int(edge_index.max().item()) + 1
        except Exception:
            pass
    return None

def _validate_edge_index(edge_index):
    try:
        import torch
    except Exception:
        return
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError("edge_index nao e torch.Tensor")
    if edge_index.dim() != 2:
        raise ValueError("edge_index deve ser tensor 2D")
    if edge_index.size(0) != 2:
        raise ValueError("edge_index deve ter shape [2, E]")
    if edge_index.dtype not in (torch.int64, torch.int32):
        raise ValueError("edge_index deve ser inteiro")
    if edge_index.numel() == 0:
        raise ValueError("edge_index vazio")

def _validate_graph_obj(obj, nodes_count_global: int | None):
    edge_index = _get_edge_index(obj)
    if edge_index is None:
        raise ValueError("obj .pt nao possui edge_index")
    _validate_edge_index(edge_index)

    num_nodes = _get_num_nodes(obj, edge_index)
    if num_nodes is None:
        raise ValueError("nao foi possivel inferir num_nodes do grafo")
    if num_nodes <= 0:
        raise ValueError("num_nodes invalido")

    try:
        import torch
        if edge_index.min().item() < 0:
            raise ValueError("edge_index possui indice negativo")
        if edge_index.max().item() >= num_nodes:
            raise ValueError("edge_index possui indice fora de num_nodes")
        if hasattr(obj, "x") and getattr(obj, "x") is not None:
            x = getattr(obj, "x")
            if isinstance(x, torch.Tensor):
                if x.dim() != 2:
                    raise ValueError("x deve ser tensor 2D [N, F]")
                if x.size(0) != num_nodes:
                    raise ValueError("x.size(0) != num_nodes")
                if torch.isnan(x).any().item():
                    raise ValueError("x possui NaN")
                if torch.isinf(x).any().item():
                    raise ValueError("x possui Inf")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise

    if nodes_count_global is not None:
        if num_nodes > nodes_count_global:
            raise ValueError("num_nodes do grafo maior que a contagem do nodes.csv")

    return num_nodes, int(edge_index.size(1))

def _parse_snapshots_txt(snap_path: Path, root: Path) -> list[Path]:
    refs = []
    lines = snap_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    base_dir = snap_path.parent
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        tokens = [t.strip() for t in s.replace("\t", " ").split(" ") if t.strip()]
        pt_tokens = [t for t in tokens if t.lower().endswith(".pt")]
        if pt_tokens:
            for t in pt_tokens:
                p = Path(t)
                if not p.is_absolute():
                    p = (root / p).resolve()
                refs.append(p)
            continue

        for t in tokens:
            direct = (base_dir / f"{t}.pt")
            if direct.exists():
                refs.append(direct.resolve())
                continue
            matches = list(base_dir.glob(f"*{t}*.pt"))
            if matches:
                refs.append(matches[0].resolve())
    return refs

def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                raise ValueError(f"JSONL invalido em {path} linha {i}")
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL linha {i} nao e objeto")
            items.append(obj)
    if not items:
        raise ValueError("instances.jsonl vazio")
    return items

def _extract_terminals(obj: dict):
    keys = ["terminals", "terminal_nodes", "terminal", "T", "query_nodes"]
    for k in keys:
        if k in obj:
            v = obj[k]
            if isinstance(v, list):
                return v, k
    return None, None

def _as_int_list(xs):
    out = []
    for v in xs:
        if isinstance(v, bool):
            raise ValueError("lista de terminais contem boolean")
        try:
            out.append(int(v))
        except Exception:
            raise ValueError(f"terminal nao-inteiro: {v}")
    return out

def validate_dataset(input_dir: Path) -> dict:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise ValueError(f"input_dir nao existe: {input_dir}")

    nodes_path = _find_file(input_dir, "nodes.csv")
    snaps_path = _find_file(input_dir, "snapshots.txt")
    inst_path = _find_file(input_dir, "instances.jsonl")

    if nodes_path is None:
        raise ValueError("nodes.csv nao encontrado")
    if snaps_path is None:
        raise ValueError("snapshots.txt nao encontrado")
    if inst_path is None:
        raise ValueError("instances.jsonl nao encontrado")

    header, node_ids, node_ids_set = _load_nodes_csv(nodes_path)

    pt_files = _iter_graph_pt_files(input_dir)
    if not pt_files:
        raise ValueError("nenhum arquivo .pt encontrado no input_dir")

    snap_refs = _parse_snapshots_txt(snaps_path, input_dir)
    missing_refs = [p for p in snap_refs if not p.exists()]
    if missing_refs:
        raise ValueError(f"snapshots.txt referencia .pt inexistente: {missing_refs[0]}")

    graphs_checked = 0
    total_nodes_in_graphs = 0
    total_edges_in_graphs = 0

    for p in pt_files[:2000]:
        obj = _torch_load(p)
        n, e = _validate_graph_obj(obj, len(node_ids))
        graphs_checked += 1
        total_nodes_in_graphs += n
        total_edges_in_graphs += e

    items = _read_jsonl(inst_path)
    inst_checked = 0
    inst_with_terminals = 0

    for obj in items[:200000]:
        terms, key = _extract_terminals(obj)
        inst_checked += 1
        if terms is None:
            continue
        inst_with_terminals += 1
        t = _as_int_list(terms)
        if not t:
            raise ValueError("instancia com lista de terminais vazia")
        if len(set(t)) != len(t):
            raise ValueError("instancia com terminais duplicados")
        for v in t:
            if v not in node_ids_set:
                raise ValueError("instancia referencia terminal que nao existe no nodes.csv")

    if inst_with_terminals == 0:
        raise ValueError("instances.jsonl nao possui nenhum campo de terminais reconhecivel")

    return {
        "input_dir": str(input_dir),
        "nodes_csv": str(nodes_path),
        "snapshots_txt": str(snaps_path),
        "instances_jsonl": str(inst_path),
        "nodes_count": len(node_ids),
        "graphs_found": len(pt_files),
        "graphs_checked": graphs_checked,
        "avg_nodes_per_graph_checked": total_nodes_in_graphs / max(1, graphs_checked),
        "avg_edges_per_graph_checked": total_edges_in_graphs / max(1, graphs_checked),
        "instances_found": len(items),
        "instances_checked": inst_checked,
        "instances_with_terminals": inst_with_terminals,
        "snapshot_pt_refs_found": len(snap_refs),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    report = validate_dataset(Path(args.input_dir))

    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validate.jsonl").write_text(out, encoding="utf-8")
    print(out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)
