"""Prefill and decode at depth: python3 longctx.py PORT [PROMPT_CHARS] [SRC_GLOB]"""
import sys, os, glob
from common import chat

port = sys.argv[1]
chars = int(sys.argv[2]) if len(sys.argv) > 2 else 220000
pattern = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/flint/llama.cpp/src/*.cpp")
text = ""
for f in sorted(glob.glob(pattern)):
    text += f"\n// FILE {f}\n" + open(f, errors="ignore").read()
    if len(text) > chars:
        break
body = {"messages": [{"role": "user", "content": text[:chars] + "\n\nSummarize what these files do, file by file."}],
        "max_tokens": 1000, "temperature": 0, "seed": 1, "cache_prompt": False}
t = chat(port, body)["timings"]
print(f'prompt {t["prompt_n"]} tok @ {t["prompt_per_second"]:.0f} tok/s | gen {t["predicted_n"]} @ {t["predicted_per_second"]:.1f} tok/s')
