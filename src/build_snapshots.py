import argparse
import bz2
import csv
from pathlib import Path
import torch

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
        print("0 0")
        return

    nodes = set()
    for p in files:
        for a, b, rel in iter_records(p):
            nodes.add(a)
            nodes.add(b)

    asn_list = sorted(nodes)
    asn2id = {asn: i for i, asn in enumerate(asn_list)}

    with open(output_dir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "asn"])
        for asn, i in asn2id.items():
            w.writerow([i, asn])

    with open(output_dir / "snapshots.txt", "w", encoding="utf-8") as f:
        for p in files:
            f.write(snapshot_name(p) + "\n")

    total_graphs = 0
    for p in files:
        src = []
        dst = []
        et = []

        for a, b, rel in iter_records(p):
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

        obj = {
            "snapshot": snapshot_name(p),
            "num_nodes": len(asn2id),
            "edge_index": edge_index,
            "edge_type": edge_type
        }

        out_path = output_dir / f"as_graph_{obj['snapshot']}.pt"
        torch.save(obj, out_path)

        total_graphs += 1
        print(obj["snapshot"], obj["num_nodes"], int(edge_index.size(1)), str(out_path))

    print("DONE", total_graphs, len(asn2id), str(output_dir))

if __name__ == "__main__":
    main()
