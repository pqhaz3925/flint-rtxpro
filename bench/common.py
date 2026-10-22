import json, urllib.request

def chat(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))
