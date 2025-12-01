import numpy as np
import faiss
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf_emb_path', type=str, required=True)
    parser.add_argument('--semantic_index_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='cf_hints.json')
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()

    print(f"Loading CF Embeddings from {args.cf_emb_path}...")
    cf_embs = np.load(args.cf_emb_path).astype(np.float32)
    faiss.normalize_L2(cf_embs) # Cosine Similarity
    
    print("Building Index...")
    index = faiss.IndexFlatIP(cf_embs.shape[1])
    index.add(cf_embs)

    print(f"Searching neighbors (Top-{args.topk})...")
    # Search K+1 because the nearest one is usually the item itself
    D, I = index.search(cf_embs, args.topk + 1)

    print(f"Loading Semantic Index from {args.semantic_index_path}...")
    with open(args.semantic_index_path, 'r') as f:
        sem_idx_map = json.load(f)

    hints = {}
    hit_count = 0
    num_items = cf_embs.shape[0]
    
    for item_id in range(num_items):
        item_id_str = str(item_id)
        neighbor_indices = I[item_id]
        
        neighbor_sids = []
        for nid in neighbor_indices:
            if nid == item_id: continue # Skip self
            nid_str = str(nid)
            if nid_str in sem_idx_map:
                # Concatenate tokens list ["<a_1>", "<b_2>"] -> "<a_1><b_2>"
                sid_str = "".join(sem_idx_map[nid_str])
                neighbor_sids.append(sid_str)
            if len(neighbor_sids) >= args.topk:
                break
        
        if neighbor_sids:
            # Format: "SID1, SID2"
            hints[item_id_str] = ", ".join(neighbor_sids)
            hit_count += 1

    with open(args.output_path, 'w') as f:
        json.dump(hints, f, indent=2)
        
    print(f"✅ Hints generated for {hit_count}/{num_items} items. Saved to {args.output_path}")

if __name__ == '__main__':
    main()
