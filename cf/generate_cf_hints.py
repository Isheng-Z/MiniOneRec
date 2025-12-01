import numpy as np
import faiss
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf_emb_path', type=str, required=True, help='LightGCN 训练出的 item embedding (.npy)')
    parser.add_argument('--semantic_index_path', type=str, required=True, help='纯语义 ID 的索引文件 (.index.json)')
    parser.add_argument('--output_path', type=str, default='cf_hints.json', help='输出的提示字典路径')
    parser.add_argument('--topk', type=int, default=3, help='每个物品找几个邻居作为提示')
    args = parser.parse_args()

    # 1. 加载协同向量
    print(f"Loading CF Embeddings from {args.cf_emb_path}...")
    cf_embs = np.load(args.cf_emb_path).astype(np.float32)
    
    # 归一化 (使用余弦相似度)
    faiss.normalize_L2(cf_embs)
    num_items = cf_embs.shape[0]

    # 2. 构建向量索引
    print("Building Faiss Index...")
    index = faiss.IndexFlatIP(cf_embs.shape[1]) # Inner Product (Normalized) = Cosine Similarity
    index.add(cf_embs)

    # 3. 搜索邻居
    # 我们搜索 K+1 个，因为第 1 个通常是物品自己
    print(f"Searching for Top-{args.topk} neighbors...")
    D, I = index.search(cf_embs, args.topk + 1)

    # 4. 加载 Semantic ID 映射 (Item ID -> "<a_x><b_y><c_z>")
    print(f"Loading Semantic Index from {args.semantic_index_path}...")
    with open(args.semantic_index_path, 'r') as f:
        # 格式: {"0": ["<a_1>", ...], "1": ...}
        sem_idx_map = json.load(f)

    # 5. 生成 Hint 字典
    # 格式: { "Item_ID_Str": "Neighbor_SID_1, Neighbor_SID_2" }
    hints = {}
    hit_count = 0
    
    for item_id in range(num_items):
        item_id_str = str(item_id)
        
        # 排除自己 (通常是 I[i][0])
        neighbor_indices = I[item_id]
        real_neighbors = []
        for nid in neighbor_indices:
            if nid != item_id:
                real_neighbors.append(nid)
            if len(real_neighbors) >= args.topk:
                break
        
        # 将邻居的 Numeric ID 转为 Semantic ID String
        neighbor_sids = []
        for nid in real_neighbors:
            nid_str = str(nid)
            if nid_str in sem_idx_map:
                # 把列表 ["<a_1>", "<b_2>"] 拼成 "<a_1><b_2>"
                sid_str = "".join(sem_idx_map[nid_str])
                neighbor_sids.append(sid_str)
        
        if neighbor_sids:
            # 用逗号连接多个邻居
            hints[item_id_str] = ", ".join(neighbor_sids)
            hit_count += 1

    # 6. 保存
    with open(args.output_path, 'w') as f:
        json.dump(hints, f, indent=2)
        
    print(f"✅ Hints generated for {hit_count}/{num_items} items.")
    print(f"Saved to {args.output_path}")

if __name__ == '__main__':
    main()