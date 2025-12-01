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

class SFTData(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        try:
            row['history_item_title'] = eval(row['history_item_title'])
        except:
            pass
        L = len(row['history_item_title']) 
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        try:
            last_history_item_id = eval(row["history_item_id"])[-1]
        except:
            last_history_item_id = None
            
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
""" 
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        
        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]
        
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Processing SFTData"):
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

class FusionSeqRecDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        with open(item_file, 'r') as f: self.item_feat = json.load(f)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids)>=3:
                self.sid2title["".join(sids[:3])] = self.item_feat[item_id]['title']
        self.get_inputs()

    def __len__(self): return len(self.data)
    def pre(self, idx):
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nCan you recommend the next item for the user based on their interaction history?\n\n"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        row = self.data.iloc[idx]
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = []
        history_str = ", ".join(history_sids)
        target_sid = str(row['item_sid'])
        target_title = self.sid2title.get(target_sid, target_sid)
        
        prompt = f"### User Input: \nThe user has sequentially interacted with items {history_str}. Can you recommend the next item for him? Tell me the title of the item\n\n### Response:\n{target_title}\n"
        tokens += self.tokenizer.encode(prompt, bos=False, eos=True)
        attention_mask = [1] * len(tokens)
        labels = tokens
        
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": attention_mask, "labels": labels}

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Processing FusionSeqRecDataset"):
            self.inputs.append(self.pre(i))
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
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nBased on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.\n\n"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        row = self.data.iloc[idx]
        try: history_titles = eval(row['history_item_title'])
        except: history_titles = []
        history_str = ", ".join([f'"{t}"' for t in history_titles])
        target_id = str(row['item_id'])
        target_sid = self.id2sid.get(target_id, target_id) + "\n"
        
        prompt = f"### User Input: \nThe user has interacted with the following items in chronological order: {history_str}. Can you predict the next item?\n\n### Response:\n{target_sid}"
        tokens += self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        labels = tokens 
        
        if len(tokens) > self.max_len:
            tokens = tokens[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": attention_mask, "labels": labels}

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Processing TitleHistory"):
            self.inputs.append(self.pre(i))
    def __getitem__(self, idx): return self.inputs[idx]

class SidItemFeatDataset(Dataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        random.seed(seed)
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.sid2title, self.title2sid = {}, {}
        
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                if len(sids) >= 3:
                    combined_sid = "".join(sids[:3])
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
        
        self.data = []
        for sid, title in self.sid2title.items():
            self.data.append({'task': 'sid2title', 'input': sid, 'output': title})
        for title, sid in self.title2sid.items():
            self.data.append({'task': 'title2sid', 'input': title, 'output': sid})
            
        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)
        if self.tokenizer is not None:
            self.get_inputs()

    def __len__(self):
        return len(self.data)
    
    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            return f"### User Input: \nWhich item has the title: {data_point['input']}?\n\n### Response:\n{data_point['output']}"
        else:
            return f"### User Input: \nWhat is the title of item \"{data_point['input']}\"?\n\n### Response:\n{data_point['output']}"
    
    def pre(self, idx):
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nAnswer the question about item identification.\n\n"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        prompt = self.generate_prompt(self.data[idx])
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        target = self.data[idx]['output'] + '\n'
        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
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
        for i in tqdm(range(len(self.data)), desc="Processing SidItemFeatDataset"):
            self.inputs.append(self.pre(i))
    
    def __getitem__(self, idx):
        return self.inputs[idx]

# =================================================================
# [重点修改] SidSFTDataset (主任务，训练用，带 Hint)
# =================================================================
class SidSFTDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False, cf_hints_path=""):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        
        # Load CF Hints
        self.cf_hints = {}
        if cf_hints_path and os.path.exists(cf_hints_path):
            print(f"🔥 [SidSFTDataset] Loading Collaborative Hints from {cf_hints_path}...")
            with open(cf_hints_path, 'r') as f:
                self.cf_hints = json.load(f)
        else:
            if cf_hints_path: print(f"⚠️ Warning: CF Hints path provided but file not found: {cf_hints_path}")

        self.get_inputs()  
    
    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"### User Input: \n{data_point['input']}\n\n### Response:\n{data_point['output']}"

    def get_history(self, row):
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = row['history_item_sid'] if isinstance(row['history_item_sid'], list) else []
        history_str = ", ".join(history_sids)
        
        # 注入 Hint
        hint_text = ""
        if 'history_item_id' in row and self.cf_hints:
            try:
                hist_ids = eval(str(row['history_item_id']))
                if hist_ids:
                    last_id = str(hist_ids[-1])
                    if last_id in self.cf_hints:
                        neighbors = self.cf_hints[last_id]
                        hint_text = f" [Hint: Users who bought the last item often also buy: {neighbors}.]"
            except: pass

        target_item = str(row['item_sid'])
        target_item_sid = row['item_sid']
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
# [重点修改] EvalSidDataset (评估用，带 Hint，Prompt格式与训练对齐)
# =================================================================
class EvalSidDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False, cf_hints_path=""):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        
        # Load Hints
        self.cf_hints = {}
        if cf_hints_path and os.path.exists(cf_hints_path):
            print(f"🔥 [EvalSidDataset] Loading Collaborative Hints from {cf_hints_path}...")
            with open(cf_hints_path, 'r') as f:
                self.cf_hints = json.load(f)
        else:
            if cf_hints_path: print(f"⚠️ Warning: Hint file not found: {cf_hints_path}")

        self.get_inputs()  
    
    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"### Example {data_point['idx']}:\n{data_point['input']} \n\n### Response:\n{data_point['output']}\n"
    
    def generate_prompt(self, data_point):
        # 统一使用 SFT 训练时的 Prompt 模板
        return f"### User Input: \n{data_point['input']}\n\n### Response:\n{data_point['output']}"

    def get_history(self, row):
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = row['history_item_sid'] if isinstance(row['history_item_sid'], list) else []
        history_str = ", ".join(history_sids)
        
        # 注入 Hint
        hint_text = ""
        if 'history_item_id' in row and self.cf_hints:
            try:
                hist_ids = eval(str(row['history_item_id']))
                if hist_ids:
                    last_id = str(hist_ids[-1])
                    if last_id in self.cf_hints:
                        neighbors = self.cf_hints[last_id]
                        hint_text = f" [Hint: Users who bought the last item often also buy: {neighbors}.]"
            except: pass

        target_item = str(row['item_sid'])
        target_item_sid = row['item_sid']
        last_history_item_sid = history_sids[-1] if history_sids else None
        
        # 关键：Evaluation 时的 Input 文本必须和 Training 时完全一致
        input_text = f"The user has interacted with items {history_str} in chronological order.{hint_text} Can you predict the next possible item that the user may expect?"
        
        return {
            "input": input_text,
            "output": target_item + '\n',
            "dedup": target_item_sid == last_history_item_sid
        }
    
    def pre(self, idx):
        instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        row = self.data.iloc[idx]
        history = self.get_history(row)
        target_item = history['output']
        history['output'] = ''
        
        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)
        
        # 评估只关注 input
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }
    
    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Processing EvalSidDataset"):
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