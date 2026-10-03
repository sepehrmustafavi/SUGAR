"""Offline stage entry point -- Algorithm 1.

Phase 1 (this batch): M1 item refinement -> cached descriptors + embeddings.
Phase 2 (next batch): M2/M3 profile building + drift detection -> snapshots.

Usage:
  python scripts/run_offline.py --dataset beauty [--phase m1|m2m3|all]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sugar.utils.config import load_config
from sugar.utils.seed import set_seed
from sugar.data.dataset import load_processed, build_item_meta_texts
from sugar.offline.llm_client import LLMClient
from sugar.offline.encoders import SentenceEncoder
from sugar.offline.item_refiner import ItemRefiner


def run_m1(cfg: dict) -> dict:
    """Refine the whole catalog once; save embeddings for M2/M3."""
    set_seed(cfg["seed"])
    name = cfg["data"]["dataset"]

    _, items, _ = load_processed(name, cfg["data"]["processed_dir"])
    texts = build_item_meta_texts(items)

    # categories per item index, for the fallback descriptor
    idx2meta = {v: k for k, v in items["item2idx"].items()}
    cats = [items["meta"][idx2meta[i]].get("categories", [])
            for i in range(len(texts))]

    llm = LLMClient(model_name=cfg["llm"]["model_name"],
                    cache_dir=f"{cfg['llm']['cache_dir']}/llm",
                    max_new_tokens=cfg["llm"]["max_new_tokens"])
    encoder = SentenceEncoder(model_name=cfg["encoder"]["model_name"],
                              cache_dir=f"{cfg['encoder']['cache_dir']}/enc")
    refiner = ItemRefiner(llm, encoder,
                          cache_dir=f"{cfg['llm']['cache_dir']}/items")

    descriptors, embeddings = refiner.refine_catalog(texts, cats)

    out_dir = Path(cfg["data"]["processed_dir"]) / name
    np.save(out_dir / "item_embeddings.npy", embeddings)
    stats = {
        "m1": refiner.stats,
        "llm": llm.stats_snapshot(),
        "n_items": len(texts),
        "emb_dim": int(embeddings.shape[1]),
    }
    with open(out_dir / "m1_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[M1] done: {stats['n_items']} items, "
          f"schema_violation_rate={refiner.stats['schema_violation_rate']}, "
          f"llm_calls={llm.stats['calls']} (cache_hits={llm.stats['cache_hits']})")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--phase", default="m1", choices=["m1", "m2m3", "all"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.phase in ("m1", "all"):
        run_m1(cfg)
    if args.phase in ("m2m3", "all"):
        # implemented in batch 6
        raise NotImplementedError("M2/M3 arrives with batch 6")


if __name__ == "__main__":
    main()