import pandas as pd
import aiohttp
import asyncio
import time
from transformers import AutoTokenizer
from tqdm import tqdm

# 配置
CSV_FILE = "TruthfulQA.csv"
API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "deepseek-r1:32b"
CONCURRENT_REQUESTS = 20  # 并发请求数量

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

async def fetch(session, question):
    global success_count, total_input_tokens, total_output_tokens, remapped_output_tokens

    # 计算输入 token 数量
    input_tokens = len(tokenizer.encode(question))
    total_input_tokens += input_tokens

    payload = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False
    }

    try:
        async with session.post(API_URL, json=payload, timeout=30) as response:
            response.raise_for_status()
            data = await response.json()
            model_answer = data.get("response", "").strip()

            # 成功请求
            success_count += 1

            # 模型输出 tokens 计算
            output_tokens = len(model_answer.split())  # 简单词计数
            total_output_tokens += output_tokens

            # 重标记后的 token 统计
            remapped_tokens = len(tokenizer.encode(model_answer))
            remapped_output_tokens += remapped_tokens

    except Exception as e:
        print(f"[失败] 请求异常: {e}")

async def main():
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, row in df.iterrows():
            question = row["Question"]
            task = fetch(session, question)
            tasks.append(task)
        await asyncio.gather(*tasks)

# 运行异步主函数
asyncio.run(main())

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


