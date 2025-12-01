import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
# 注意：这里需要导入 ConstrainedLogitsProcessor，假设它在当前目录或父目录
# 如果报错找不到，可能需要 sys.path.append
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LogitProcessor import ConstrainedLogitsProcessor # 确保 LogitProcessor.py 在 dirA 里
from data import EvalSidDataset  
import random
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(
    base_model: str = "",
    test_data_path: str = "",
    info_file: str = "",
    category: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 20,
    cf_hints_path: str = "", 
):
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Category: {category}")
    print(f"Loading Hints from: {cf_hints_path}")

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except:
        print("⚠️ Warning: Could not load tokenizer from model path. Trying default Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # 构建约束字典 (Constrained Decoding)
    with open(info_file, 'r') as f:
        info = f.readlines()
        semantic_ids = [line.split('\t')[0].strip() for line in info]
        # 添加 Response 模板前缀模拟
        info_semantic = [f'''### Response:\n{_}\n''' for _ in semantic_ids] 

    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
    
    # Qwen 的 prefix_index 可能需要根据实际情况调整，这里沿用原逻辑
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        # 模拟原代码逻辑构建 Trie
        ID_with_eos = ID + [tokenizer.eos_token_id] # 显式加 EOS，原代码逻辑似乎隐含了
        # 原代码 evaluate.py 逻辑：
        # ID.append(tokenizer.eos_token_id)
        # for i in range(prefix_index, len(ID)):
        #     ...
        
        # 让我们严谨一点，复制原逻辑
        curr_ID = list(ID) # copy
        curr_ID.append(tokenizer.eos_token_id)
        
        for i in range(prefix_index, len(curr_ID)):
            if i == prefix_index:
                hash_number = get_hash(curr_ID[:i])
            else:
                hash_number = get_hash(curr_ID[prefix_index:i]) # 注意这里是相对索引
            
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(curr_ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    # ========================================================
    # [核心修复] 定义真正的约束函数
    # ========================================================
    def prefix_allowed_tokens_fn(batch_id, input_ids_list):
        # LogitProcessor 传进来的 input_ids_list 已经是根据 count 截取过的 hash_key
        # 直接拿来查 hash_dict 即可
        hash_number = get_hash(input_ids_list)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return [] # 如果查不到，返回空 list，LogitProcessor 会跳过约束(continue)
    # ========================================================

    # 实例化 Dataset
    val_dataset = EvalSidDataset(
        train_file=test_data_path, 
        tokenizer=tokenizer, 
        max_len=2048, 
        category=category, 
        test=True, 
        K=K, 
        seed=seed,
        cf_hints_path=cf_hints_path 
    )
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    def evaluate_batch(batch_encodings):
        maxLen = max([len(_["input_ids"]) for _ in batch_encodings])
        input_ids = []
        attention_mask = []

        for enc in batch_encodings:
            p_len = maxLen - len(enc["input_ids"])
            input_ids.append([tokenizer.pad_token_id] * p_len + enc["input_ids"])
            attention_mask.append([0] * p_len + [1] * len(enc["input_ids"]))

        input_tensor = torch.tensor(input_ids).to(device)
        mask_tensor = torch.tensor(attention_mask).to(device)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens
        )
        
        # 传入真正的函数
        clp = ConstrainedLogitsProcessor(
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            base_model=base_model,
        )
        
        logits_processor = LogitsProcessorList([clp])

        with torch.no_grad():
            outputs = model.generate(
                input_tensor,
                attention_mask=mask_tensor,
                generation_config=generation_config,
                logits_processor=logits_processor
            )
        
        # Decode
        generated_sequences = outputs[:, maxLen:]
        decoded = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        
        batch_results = []
        for i in range(0, len(decoded), num_beams):
            batch_results.append(decoded[i : i + num_beams])
        return batch_results

    from tqdm import tqdm
    all_predictions = []
    
    batch_chunks = [encodings[i:i + batch_size] for i in range(0, len(encodings), batch_size)]
    
    for batch in tqdm(batch_chunks, desc="Evaluating"):
        results = evaluate_batch(batch)
        all_predictions.extend(results)

    for i, res in enumerate(all_predictions):
        test_data[i]["predict"] = res
        if 'dedup' in test_data[i]: del test_data[i]['dedup']

    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"Results saved to {result_json_data}")

if __name__ == '__main__':
    fire.Fire(main)