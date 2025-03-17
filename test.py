import pandas as pd
import requests
import json
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# 配置
CSV_FILE = "TruthfulQA.csv"
API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen:32b"

# 加载数据
df = pd.read_csv(CSV_FILE, sep='\t')

# 加载 tokenizer（这里假设用 GPT2 的，你可以替换成 deepseek-r1 的 tokenizer）
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 性能统计指标
success_count = 0
total_input_tokens = 0
total_output_tokens = 0
remapped_output_tokens = 0

start_time = time.time()



# 遍历数据集请求
for idx, row in tqdm(df.iterrows(), total=len(df)):
    
    
    question = row["Question"]
    
    # 计算输入 token 数量
    input_tokens = len(re.findall(r'\b\w+\b|[\W_]', question))
    total_input_tokens += input_tokens

    
    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        
        data = response.json()
        model_answer = data.get("response", "").strip()
        
        # 成功请求
        success_count += 1
        
        # 模型输出 tokens 计算
        output_tokens = len(model_answer.split())  # 简单词计数
        total_output_tokens += output_tokens
        
        
        # 重标记后的 token 统计
        remapped_tokens = len(re.findall(r'\b\w+\b|[\W_]', model_answer))
        remapped_output_tokens += remapped_tokens
        
    except Exception as e:
        print(f"[失败] 第 {idx+1} 个问题 请求异常: {e}")
        continue

end_time = time.time()
elapsed_time = end_time - start_time

# 吞吐量计算
req_per_sec = success_count / elapsed_time
input_tok_per_sec = total_input_tokens / elapsed_time
output_tok_per_sec = total_output_tokens / elapsed_time

# 输出统计结果
print("\n===== 基准测试结果 =====")
print(f"成功请求数: {success_count}")
print(f"基准测试持续时间 (秒): {elapsed_time:.2f}")
print(f"总输入 tokens 数量: {total_input_tokens}")
print(f"总生成 tokens 数量: {total_output_tokens}")
print(f"总生成 tokens 数量（重标记后）: {remapped_output_tokens}")
print(f"请求吞吐量 (req/s): {req_per_sec:.2f}")
print(f"输入 token 吞吐量 (tok/s): {input_tok_per_sec:.2f}")
print(f"输出 token 吞吐量 (tok/s): {output_tok_per_sec:.2f}")


