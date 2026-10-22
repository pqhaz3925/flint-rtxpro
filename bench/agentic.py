"""Free text vs. 'edit this file' (output repeats input, where n-gram speculation pays off).
python3 agentic.py PORT [FILE_TO_EDIT]"""
import sys, os
from common import chat

port = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/flint/llama.cpp/src/models/flint.cpp")
code = open(path).read()
tasks = {
    "essay": "Explain how TCP congestion control works, step by step.",
    "edit": f"Here is a file:\n```\n{code}\n```\nRename the variable `cur` to `x` everywhere and output the full updated file in a single code block. No explanations.",
}
for name, p in tasks.items():
    t = chat(port, {"messages": [{"role": "user", "content": p}], "max_tokens": 6000, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1})["timings"]
    extra = f', drafted {t["draft_n"]} accepted {t["draft_n_accepted"]}' if "draft_n" in t else ""
    print(f'{name}: {t["predicted_n"]} tok @ {t["predicted_per_second"]:.1f} tok/s{extra}', flush=True)
