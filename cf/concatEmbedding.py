import numpy as np
import argparse
from sklearn.preprocessing import normalize


def main():
    parser = argparse.ArgumentParser(description="Merge Semantic Embeddings with Collaborative Embeddings")
    parser.add_argument('--semantic_emb', type=str, required=True, help='Path to Qwen/Semantic .npy file')
    parser.add_argument('--cf_emb', type=str, required=True, help='Path to Collaborative Filter .npy file')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for merged .npy file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for CF embedding (default: 1.0)')
    args = parser.parse_args()

    print(f"加载语义向量: {args.semantic_emb}")
    sem_emb = np.load(args.semantic_emb)

    print(f"加载协同向量: {args.cf_emb}")
    cf_emb = np.load(args.cf_emb)

    # 1. 维度检查与对齐
    if sem_emb.shape[0] != cf_emb.shape[0]:
        print(f"⚠️ 警告: 物品数量不一致! Semantic: {sem_emb.shape[0]}, CF: {cf_emb.shape[0]}")
        min_len = min(sem_emb.shape[0], cf_emb.shape[0])
        print(f"自动截断至: {min_len}")
        sem_emb = sem_emb[:min_len]
        cf_emb = cf_emb[:min_len]

    # 2. 归一化 (非常重要，防止某一方数值过大主导距离计算)
    print("正在进行 L2 归一化...")
    sem_emb = normalize(sem_emb, axis=1)
    cf_emb = normalize(cf_emb, axis=1)

    # 3. 加权与拼接
    # 也可以尝试加权: merged = np.concatenate([sem_emb, args.alpha * cf_emb], axis=1)
    print(f"拼接向量 (CF权重: {args.alpha})...")
    merged_emb = np.concatenate([sem_emb, args.alpha * cf_emb], axis=1)

    # 4. 保存
    np.save(args.output_path, merged_emb)
    print(f"✅ 融合完成! 保存至: {args.output_path}")
    print(f"最终形状: {merged_emb.shape}")


if __name__ == '__main__':
    main()