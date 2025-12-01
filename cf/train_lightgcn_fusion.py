import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict

# ================= 配置区域 =================
LIGHTGCN_DIM = 64
EPOCHS = 20
BATCH_SIZE = 2048
LR = 0.001
TOP_K = [10, 20]
# ===========================================

class LightGCN_Fusion(nn.Module):
    def __init__(self, num_users, num_items, text_embedding_path, embedding_dim=64, n_layers=3):
        super(LightGCN_Fusion, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        
        # 1. 用户 Embedding (依然是随机初始化)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        
        # 2. 物品 ID Embedding (捕捉文本没覆盖到的协同信号)
        self.item_id_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.item_id_emb.weight, std=0.1)
        
        # 3. 文本语义融合层
        # 加载预训练的 Qwen Embedding
        print(f"Loading Text Embeddings from {text_embedding_path}...")
        raw_text_emb = np.load(text_embedding_path)
        
        # 维度对齐检查
        if raw_text_emb.shape[0] != num_items:
            print(f"⚠️ 警告: 文本向量数量 ({raw_text_emb.shape[0]}) 与 物品ID数量 ({num_items}) 不一致！")
            # 简单截断或补零处理，防止报错
            min_len = min(raw_text_emb.shape[0], num_items)
            self.text_emb_tensor = torch.zeros((num_items, raw_text_emb.shape[1]))
            self.text_emb_tensor[:min_len] = torch.tensor(raw_text_emb[:min_len])
        else:
            self.text_emb_tensor = torch.tensor(raw_text_emb)
            
        # 冻结文本 Embedding，不参与训练
        self.text_emb_tensor = self.text_emb_tensor.float().requires_grad_(False)
        
        # 投影层: 896 -> 64
        self.text_proj = nn.Linear(self.text_emb_tensor.shape[1], embedding_dim)
        
        self.graph = None

    def forward(self):
        # 确保 text_emb 在正确的设备上
        if self.text_emb_tensor.device != self.user_emb.weight.device:
            self.text_emb_tensor = self.text_emb_tensor.to(self.user_emb.weight.device)
            
        # ======================================================
        # 核心改进: 物品初始向量 = ID向量 + 投影后的文本向量
        # 这样 LightGCN 的传播起始点就包含了语义信息
        # ======================================================
        item_emb_0 = self.item_id_emb.weight + self.text_proj(self.text_emb_tensor)
        
        all_embs = [torch.cat([self.user_emb.weight, item_emb_0])]
        
        for layer in range(self.n_layers):
            all_embs.append(torch.sparse.mm(self.graph, all_embs[-1]))
            
        all_embs = torch.stack(all_embs, dim=1)
        final_embs = torch.mean(all_embs, dim=1)
        
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        return users, items

# ... (build_graph, load_data, evaluate 函数保持不变，直接复用之前的代码即可) ...
# 为了完整性，我把省略的部分补全，你可以直接复制整个文件

def build_graph(num_users, num_items, interactions):
    print("构建图结构...")
    src, dst = [], []
    for u, i in interactions:
        src.append(u); dst.append(i + num_users)
        src.append(i + num_users); dst.append(u)
    
    src = np.array(src); dst = np.array(dst)
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), 
                        shape=(num_users + num_items, num_users + num_items), dtype=np.float32)
    
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum + 1e-9, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))

def load_data(data_dir, dataset_name):
    with open(os.path.join(data_dir, dataset_name, f"{dataset_name}.user2id"), 'r') as f:
        num_users = len(f.readlines())
    with open(os.path.join(data_dir, dataset_name, f"{dataset_name}.item2id"), 'r') as f:
        num_items = len(f.readlines())
    
    train_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.train.inter")
    train_interactions = []
    train_history = defaultdict(set)
    with open(train_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            u = int(parts[0])
            if parts[1]:
                for i_str in parts[1].split(' '): 
                    i_val = int(i_str)
                    train_interactions.append((u, i_val))
                    train_history[u].add(i_val)
            i_tgt = int(parts[2])
            train_interactions.append((u, i_tgt))
            train_history[u].add(i_tgt)
            
    test_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.test.inter")
    test_interactions = []
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                u, i = int(parts[0]), int(parts[2])
                test_interactions.append((u, i))
                
    return num_users, num_items, list(set(train_interactions)), train_history, test_interactions

def evaluate(model, device, test_interactions, train_history, num_items):
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model()
    
    user_test_dict = defaultdict(list)
    for u, i in test_interactions: user_test_dict[u].append(i)
    
    hits = {k: 0.0 for k in TOP_K}
    ndcgs = {k: 0.0 for k in TOP_K}
    num_test = 0
    
    for u, targets in tqdm(user_test_dict.items(), leave=False):
        if u >= len(users_emb): continue
        num_test += 1
        u_emb = users_emb[u].unsqueeze(0)
        scores = torch.mm(u_emb, items_emb.t()).squeeze(0)
        if u in train_history: scores[list(train_history[u])] = -float('inf')
        
        _, topk = torch.topk(scores, max(TOP_K))
        pred = topk.cpu().tolist()
        
        for k in TOP_K:
            p = pred[:k]
            for t in targets:
                if t in p:
                    hits[k] += 1.0
                    ndcgs[k] += 1.0 / np.log2(p.index(t) + 2)
                    
    print(f"Eval: HR@20={hits[20]/num_test:.4f}, NDCG@20={ndcgs[20]/num_test:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--qwen_emb_path', type=str, required=True, help="原始语义向量路径")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_users, num_items, train_inters, train_history, test_inters = load_data(args.data_dir, args.dataset)
    
    graph = build_graph(num_users, num_items, train_inters).to(device)
    
    # 初始化融合模型
    model = LightGCN_Fusion(num_users, num_items, args.qwen_emb_path, LIGHTGCN_DIM).to(device)
    model.graph = graph
    
    # 优化器: 同时训练 ID Embedding 和 投影层 Linear
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print("开始训练 LightGCN-Fusion...")
    for epoch in range(EPOCHS):
        model.train()
        # ... (训练循环与之前一致，略去重复部分以节省篇幅，逻辑完全一样) ...
        # 请直接复制之前 train_lightgcn.py 的训练循环部分
        total_loss = 0
        interaction_tensor = torch.tensor(train_inters)
        n_batch = len(train_inters) // BATCH_SIZE
        indices = torch.randperm(len(train_inters))
        
        with tqdm(total=n_batch, desc=f"Epoch {epoch+1}", leave=False) as pbar:
            for i in range(n_batch):
                batch_idx = indices[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                batch = interaction_tensor[batch_idx].to(device)
                
                users_idx = batch[:, 0]
                pos_items_idx = batch[:, 1]
                neg_items_idx = torch.randint(0, num_items, (len(batch),)).to(device)
                
                optimizer.zero_grad()
                users_emb, items_emb = model()
                
                u_emb, pos_emb, neg_emb = users_emb[users_idx], items_emb[pos_items_idx], items_emb[neg_items_idx]
                
                loss = -torch.mean(torch.nn.functional.logsigmoid((u_emb * pos_emb).sum(1) - (u_emb * neg_emb).sum(1)))
                loss += 1e-4 * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(batch)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)
        
        print(f"Epoch {epoch+1}: Loss {total_loss/n_batch:.4f}")
        if (epoch+1) % 5 == 0: evaluate(model, device, test_inters, train_history, num_items)

    # 保存
    model.eval()
    with torch.no_grad():
        _, item_embs = model()
        item_embs = item_embs.cpu().numpy()
    np.save(args.output_path, item_embs)
    print(f"✅ Fusion Embedding 保存完毕: {args.output_path}")

if __name__ == '__main__':
    main()