"""Quick test script for the /api/chat endpoint."""
import json
import sys
import time
import urllib.request

BASE = "http://192.168.66.190:8765"
text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hello"

print(f"Sending: {text}")
t0 = time.time()
req = urllib.request.Request(
    f"{BASE}/api/chat",
    data=json.dumps({"text": text}).encode("utf-8"),
    headers={"Content-Type": "application/json; charset=utf-8"},
)
resp = urllib.request.urlopen(req, timeout=60)
raw = resp.read()
data = json.loads(raw)
dt = time.time() - t0

# Write to file (Windows console mangles UTF-8)
out = f"Time: {dt:.1f}s\nReply: {data.get('reply', 'NONE')}\n"
outfile = "D:/inovxio/tools/askme/.tmp/chat_test.txt"
with open(outfile, "w", encoding="utf-8") as f:
    f.write(out)
print(f"Done in {dt:.1f}s — see {outfile}")
