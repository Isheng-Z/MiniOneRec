import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

def process_csv(file_path, hints, output_path, is_train=False):
    print(f"处理文件: {file_path} | 模式: {'[训练-去泄露]' if is_train else '[测试-全保留]'}")
    df = pd.read_csv(file_path)
    new_hints = []
    leak_cnt = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        target = str(row['item_sid']).strip()
        hint_str = ""
        
        # 获取 History 最后一个物品 ID
        last_id = None
        if 'history_item_id' in row:
            try:
                h_ids = eval(str(row['history_item_id']))
                if h_ids: last_id = str(h_ids[-1])
            except: pass
        
        if last_id and last_id in hints:
            # 原始邻居列表
            neighbors = hints[last_id].split(', ')
            
            if is_train:
                # 【训练集】：剔除 Target
                safe_neighbors = [n for n in neighbors if n.strip() != target]
                if len(safe_neighbors) < len(neighbors):
                    leak_cnt += 1
            else:
                # 【测试集】：保留所有 (模拟真实推荐)
                safe_neighbors = neighbors
            
            if safe_neighbors:
                hint_content = ", ".join(safe_neighbors)
                hint_str = f" [Hint: Users who bought the last item often also buy: {hint_content}.]"
        
        new_hints.append(hint_str)

    df['safe_hint'] = new_hints
    df.to_csv(output_path, index=False)
    if is_train:
        print(f"🛡️  已清洗 {leak_cnt} 条泄露数据！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='./data/sft_ready')
    parser.add_argument('--dst_dir', type=str, default='./data/sft_safe')
    parser.add_argument('--hints', type=str, default='dirA/cf_hints.json')
    args = parser.parse_args()

    with open(args.hints) as f:
        hints = json.load(f)

    # 遍历目录处理
    for split, is_train in [('train', True), ('valid', False), ('test', False)]:
        src = os.path.join(args.src_dir, split)
        dst = os.path.join(args.dst_dir, split)
        for fname in os.listdir(src):
            if fname.endswith('.csv'):
                process_csv(os.path.join(src, fname), hints, os.path.join(dst, fname), is_train)

if __name__ == '__main__':
    main()