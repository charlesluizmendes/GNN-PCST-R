import argparse
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm

def list_graphs(input_dir):
    d = Path(input_dir)
    return sorted(d.glob("as_graph_*.pt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    graphs = list_graphs(args.input_dir)
    if not graphs:
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "instances.jsonl"

    random.seed(42)
    total_instances = 0

    for gp in graphs:
        g = torch.load(gp, map_location="cpu")
        snap = str(g.get("snapshot", ""))
        server_id = int(g.get("server_id", 0))
        node_types = g.get("node_types", {})
        
        antennas = [node_id for node_id, tipo in node_types.items() if tipo == "antena"]
        
        if len(antennas) == 0:
            continue
        
        num_instances = len(antennas)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for inst_id in tqdm(
                range(num_instances),
                desc="generate_instances: escrevendo instances.jsonl",
                unit="instância"
            ):
                k = random.randint(10, 30)
                terminals = random.sample(antennas, k=min(k, len(antennas)))
                
                rec = {
                    "id": total_instances,
                    "snapshot": snap,
                    "root": server_id,
                    "terminals": terminals
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_instances += 1
        
        print()

if __name__ == "__main__":
    main()