"""Short-context decode speed: python3 decode.py PORT [RUNS]"""
import sys
from common import chat

port = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
prompts = ["Write a detailed essay about the history of compilers.",
           "Implement a red-black tree in C++ with insert and delete, with comments.",
           "Explain how TCP congestion control works, step by step."]
res = []
for i in range(n):
    t = chat(port, {"messages": [{"role": "user", "content": prompts[i % 3]}], "max_tokens": 1500, "temperature": 0, "seed": 1})["timings"]
    res.append(t["predicted_per_second"])
    print(f'#{i}: {t["predicted_n"]} tok @ {t["predicted_per_second"]:.1f} tok/s ({t["predicted_per_token_ms"]:.2f} ms/tok)', flush=True)
print("avg", round(sum(res) / len(res), 1), "tok/s")
