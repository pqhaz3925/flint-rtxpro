"""Greedy output of two servers must match (patched vs stock): python3 same_output.py PORT_A PORT_B"""
import sys
from common import chat

def run(port):
    m = chat(port, {"messages": [{"role": "user", "content": "Implement a red-black tree in C++ with insert and delete."}],
                    "max_tokens": 800, "temperature": 0, "seed": 1})["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")

a, b = run(sys.argv[1]), run(sys.argv[2])
n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print("identical" if a == b else f"diverge at char {n} of {len(a)}")
