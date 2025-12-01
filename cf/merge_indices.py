import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic_index', type=str, required=True, help='Path to semantic index json (e.g., Depth=3)')
    parser.add_argument('--collaborative_index', type=str, required=True, help='Path to collaborative index json (e.g., Depth=1)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final merged index json')
    args = parser.parse_args()

    print(f"Loading Semantic Index: {args.semantic_index}")
    with open(args.semantic_index, 'r') as f:
        sem_idx = json.load(f)

    print(f"Loading Collaborative Index: {args.collaborative_index}")
    with open(args.collaborative_index, 'r') as f:
        col_idx = json.load(f)

    merged_idx = {}
    
    # 统计信息
    common_keys = set(sem_idx.keys()).intersection(set(col_idx.keys()))
    print(f"Semantic items: {len(sem_idx)}, Collaborative items: {len(col_idx)}")
    print(f"Overlapping items: {len(common_keys)}")

    # 遍历所有语义 ID (以语义 ID 为主，保证覆盖率)
    for item_id, s_tokens in sem_idx.items():
        # s_tokens 样例: ["<a_10>", "<b_5>", "<c_3>"]
        
        c_tokens = []
        if item_id in col_idx:
            raw_c_tokens = col_idx[item_id] # 样例: ["<a_8>"]
            
            # 【关键步骤】重命名协同 Token，防止冲突
            # 将 <a_8> 变成 <cf_a_8>
            # 这样 LLM 就能区分这是“协同信号”而不是“语义信号的第一层”
            c_tokens = [t.replace("<", "<cf_") for t in raw_c_tokens]
        else:
            # 冷启动/无交互物品：使用特定的 UNK 符号
            # 假设协同 ID 深度为 1
            c_tokens = ["<cf_unk>"]

        # 拼接：语义在前，协同在后 (你也可以尝试反过来)
        # 结果: ["<a_10>", "<b_5>", "<c_3>", "<cf_a_8>"]
        merged_idx[item_id] = s_tokens + c_tokens

    # 保存
    with open(args.output_path, 'w') as f:
        json.dump(merged_idx, f, indent=2)
    
    print(f"✅ Merged Index 已保存至: {args.output_path}")
    
    # 打印一个样例供检查
    sample_id = list(merged_idx.keys())[0]
    print(f"Sample Item {sample_id}: {merged_idx[sample_id]}")

if __name__ == '__main__':
    main()