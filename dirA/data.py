import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.tokenizer.encode(s)
        if self.bos_id is not None:
            while len(t) > 0 and t[0] == self.bos_id: t = t[1:]
        if self.eos_id is not None:
            while len(t) > 0 and t[-1] == self.eos_id: t = t[:-1]
        if bos and self.bos_id is not None: t = [self.bos_id] + t
        if eos and self.eos_id is not None: t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

# =================================================================
# 1. 核心推荐任务 (修改：加入 Hint Dropout)
# =================================================================
class SidSFTDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False, cf_hints_path="", hint_dropout_rate=0.3):
        """
        hint_dropout_rate: 训练时随机丢弃 Hint 的概率 (默认 0.3 即 30%)
        """
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.hint_dropout_rate = hint_dropout_rate
        
        # 加载 Hints
        self.cf_hints = {}
        if cf_hints_path and os.path.exists(cf_hints_path):
            print(f"🔥 [SidSFTDataset] Loading Collaborative Hints from {cf_hints_path}...")
            with open(cf_hints_path, 'r') as f:
                self.cf_hints = json.load(f)
        else:
            if cf_hints_path: print(f"⚠️ Warning: CF Hints provided but file not found: {cf_hints_path}")
            
        if not self.test and cf_hints_path:
             print(f"🎲 [SidSFTDataset] Hint Dropout Enabled: p={self.hint_dropout_rate}")

        self.get_inputs()  
    
    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"### User Input: \n{data_point['input']}\n\n### Response:\n{data_point['output']}"

    def get_history(self, row):
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = row['history_item_sid'] if isinstance(row['history_item_sid'], list) else []
        history_str = ", ".join(history_sids)
        
        target_item_sid = str(row['item_sid'])
        hint_text = ""
        
        # 1. 查找并清洗 Hint (训练集剔除 Target)
        if 'history_item_id' in row and self.cf_hints:
            try:
                hist_ids = eval(str(row['history_item_id']))
                if hist_ids:
                    last_id = str(hist_ids[-1])
                    if last_id in self.cf_hints:
                        raw_neighbors_str = self.cf_hints[last_id]
                        neighbor_list = [n.strip() for n in raw_neighbors_str.split(', ')]
                        
                        # 训练模式：剔除 Target
                        if not self.test:
                            clean_neighbors = [n for n in neighbor_list if n != target_item_sid]
                        else:
                            clean_neighbors = neighbor_list
                        
                        if clean_neighbors:
                            neighbors_str = ", ".join(clean_neighbors)
                            hint_text = f" [Hint: Users who bought the last item often also buy: {neighbors_str}.]"
            except: pass

        # ====================================================
        # [核心逻辑] Hint Dropout
        # 即使查到了 Hint，也有 30% 概率强制丢弃，逼迫模型看 History
        # ====================================================
        if not self.test and hint_text and random.random() < self.hint_dropout_rate:
            hint_text = ""

        target_item = str(row['item_sid'])
        last_history_item_sid = history_sids[-1] if history_sids else None
        
        input_text = f"The user has interacted with items {history_str} in chronological order.{hint_text} Can you predict the next possible item that the user may expect?"
        
        return {
            "input": input_text,
            "output": target_item + "\n",
            "history_str": history_str,
            "dedup": target_item_sid == last_history_item_sid
        }
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        row = self.data.iloc[idx]
        history = self.get_history(row)
        target_item = history['output']
        
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {"input_ids": tokens, "attention_mask": attention_mask}    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_len = len(tokens)
        tokens += golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_len + tokens[input_len:]
        
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]
            
        return {"input_ids": tokens, "attention_mask": attention_mask, "labels": labels}
    
    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Processing SidSFTDataset"):
            self.inputs.append(self.pre(i))
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

# =================================================================
# 2. EvalSidDataset (评估用，自动继承，无 Dropout)
# =================================================================
class EvalSidDataset(SidSFTDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False, cf_hints_path=""):
        # 强制 test=True，hint_dropout=0.0
        super().__init__(train_file, tokenizer, max_len, sample, True, seed, category, K, dedup, cf_hints_path, hint_dropout_rate=0.0)

# =================================================================
# 3-6. 原版辅助任务类 (保持原样，无 Hint 逻辑)
# =================================================================
class SidItemFeatDataset(Dataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        random.seed(seed)
        with open(item_file, 'r') as f: self.item_feat = json.load(f)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.sid2title, self.title2sid = {}, {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids)>=3:
                combined_sid = "".join(sids[:3])
                title = self.item_feat[item_id]['title']
                self.sid2title[combined_sid] = title
                self.title2sid[title] = combined_sid
        self.data = []
        for sid, title in self.sid2title.items():
            self.data.append({'task': 'sid2title', 'input': sid, 'output': title})
        for title, sid in self.title2sid.items():
            self.data.append({'task': 'title2sid', 'input': title, 'output': sid})
        if sample > 0: self.data = random.sample(self.data, sample)
        if self.tokenizer: self.get_inputs()
    def __len__(self): return len(self.data)
    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            return f"### User Input: \nWhich item has the title: {data_point['input']}?\n\n### Response:\n{data_point['output']}"
        else:
            return f"### User Input: \nWhat is the title of item \"{data_point['input']}\"?\n\n### Response:\n{data_point['output']}"
    def pre(self, idx):
        instruction = "Below is an instruction... Answer the question about item identification.\n\n"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        prompt = self.generate_prompt(self.data[idx])
        tokens += self.tokenizer.encode(prompt, bos=False, eos=False)
        target = self.data[idx]['output'] + '\n'
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_len = len(tokens)
        tokens += golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_len + tokens[input_len:]
        if len(tokens) > self.max_len: tokens = tokens[-self.max_len:]; attention_mask = attention_mask[-self.max_len:]; labels = labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": attention_mask, "labels": labels}
    def get_inputs(self):
        self.inputs = [self.pre(i) for i in tqdm(range(len(self.data)), desc="Processing Feat Task")]
    def __getitem__(self, idx): return self.inputs[idx]

class FusionSeqRecDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        with open(item_file, 'r') as f: self.item_feat = json.load(f)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids)>=3: self.sid2title["".join(sids[:3])] = self.item_feat[item_id]['title']
        self.get_inputs()
    def __len__(self): return len(self.data)
    def pre(self, idx):
        instruction = "Below is an instruction... Can you recommend the next item for the user based on their interaction history?\n\n"
        tokens = self.tokenizer.encode(instruction, True, False)
        row = self.data.iloc[idx]
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = []
        history_str = ", ".join(history_sids)
        target_sid = str(row['item_sid'])
        target_title = self.sid2title.get(target_sid, target_sid)
        prompt = f"### User Input: \nThe user has sequentially interacted with items {history_str}. Can you recommend the next item for him? Tell me the title of the item\n\n### Response:\n{target_title}\n"
        tokens += self.tokenizer.encode(prompt, False, True)
        labels = tokens 
        if len(tokens) > self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}
    def get_inputs(self):
        self.inputs = [self.pre(i) for i in tqdm(range(len(self.data)), desc="Processing FusionSeq")]
    def __getitem__(self, idx): return self.inputs[idx]

class SFTData(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.category = category
        self.instructs = [f"Given a list of {category}...", f"Considering the {category}...", f"Based on the user's current gaming preference...", f"Reflecting on the {category}...", f"In light of the recent gaming enjoyment...", f"Taking into account the {category}...", f"Given the user's newfound enjoyment...", f"In response to the user's recent fondness...", f"With respect to the {category}...", f"Bearing in mind the {category}...", f"In relation to the user's recent entertainment with a given {category}..." ]
        self.get_inputs()
    def __len__(self): return len(self.data)
    def pre(self, idx):
        row = self.data.iloc[idx]
        try: history = eval(row['history_item_title'])
        except: history = []
        hist_str = ", ".join([f'"{t}"' for t in history])
        target = f'"{row["item_title"]}"\n'
        prompt = f"### User Input: \nThe user has palyed the following {self.category}s before: {hist_str}\n\n### Response:\n{target}"
        instruction = f"Below is an instruction... \n\n### Instruction:\n{self.instructs[random.randint(0, len(self.instructs)-1)]}\n"
        tokens = self.tokenizer.encode(instruction, True, False) + self.tokenizer.encode(prompt, False, False)
        labels = tokens 
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}
    def get_inputs(self):
        self.inputs = [self.pre(i) for i in tqdm(range(len(self.data)), desc="Processing SFTData")]
    def __getitem__(self, idx): return self.inputs[idx]

class TitleHistory2SidSFTDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3: self.id2sid[item_id] = "".join(sids[:3])
        self.get_inputs()
    def __len__(self): return len(self.data)
    def pre(self, idx):
        row = self.data.iloc[idx]
        try: history = eval(row['history_item_title'])
        except: history = []
        hist_str = ", ".join([f'"{t}"' for t in history])
        target_id = str(row['item_id'])
        target_sid = self.id2sid.get(target_id, target_id) + "\n"
        prompt = f"### User Input: \nThe user has interacted... {hist_str}... predict the semantic ID...\n\n### Response:\n{target_sid}"
        instruction = "Below is an instruction... Based on the user's historical interaction with item titles...\n\n"
        tokens = self.tokenizer.encode(instruction, True, False) + self.tokenizer.encode(prompt, False, False)
        labels = tokens
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}
    def get_inputs(self):
        self.inputs = [self.pre(i) for i in tqdm(range(len(self.data)), desc="Processing TitleHistory")]
    def __getitem__(self, idx): return self.inputs[idx]