# 🦙 Llama 3 8B — 3-Instance Pipeline on Kaggle

## Architecture Summary

```
Client → [Stage 0: embed + layers 0–10] →(ngrok)→ [Stage 1: layers 11–21] →(ngrok)→ [Stage 2: layers 22–31 + head] → token
              ↑______________________________________________repeat per token______________________________________________↑
```

Each stage runs on a separate Kaggle notebook with a T4 GPU. Hidden states (8 KB per token) are passed
between stages over ngrok tunnels. Only Stage 2 produces a token — it returns `token_id` back through
Stage 1 → Stage 0 → client.

---

## Prerequisites

### HuggingFace Access
- Request access to `meta-llama/Meta-Llama-3-8B` at [huggingface.co/meta-llama](https://huggingface.co/meta-llama)
- Create a token with `read` permission at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### ngrok Accounts
ngrok free tier allows **1 active tunnel per account**. You need **3 separate ngrok accounts**, one per instance.
- Sign up at [ngrok.com](https://ngrok.com) — free tier is sufficient
- Copy the auth token from each account's dashboard

### Kaggle Instances
- Open **3 separate Kaggle notebooks**
- For each: Settings → Accelerator → **GPU T4 x1**
- Upload `stage0_server.py`, `stage1_server.py`, `stage2_server.py` as a Kaggle dataset,
  or paste the file contents directly into a notebook cell

---

## Step-by-Step Boot Order

### ⚠️ CRITICAL: Always boot in this order — Stage 2 → Stage 1 → Stage 0 → Client.
### Each stage needs the URL of the next one before it can start.

---

### Stage 2 (Kaggle Instance 2) — Boot First

**Notebook setup cells:**

```python
# Cell 1 — Install
!pip install -q transformers accelerate flask pyngrok torch sentencepiece

# Cell 2 — HuggingFace auth (need access to meta-llama/Meta-Llama-3-8B)
from huggingface_hub import login
login("hf_YOUR_TOKEN_HERE")   # or use Kaggle secrets

# Cell 3 — ngrok token for Instance 2 (must be a separate ngrok account)
import os
os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_2"

# Cell 4 — Run Stage 2 server
exec(open("stage2_server.py").read())
# OR if uploaded as a Kaggle dataset:
# !python /kaggle/input/llm-pipeline/stage2_server.py
```

**Stage 2 will print:**
```
============================================================
[Stage 2] PUBLIC URL: https://aaa111.ngrok-free.app
  → Copy this URL into stage1_server.py as STAGE2_URL
  → e.g.  STAGE2_URL = "https://aaa111.ngrok-free.app"
============================================================
```

**Copy that URL. You will paste it into Stage 1 next.**

---

### Stage 1 (Kaggle Instance 1) — Boot Second

```python
# Cell 1 — Install
!pip install -q transformers accelerate flask pyngrok torch sentencepiece requests

# Cell 2 — HuggingFace auth
from huggingface_hub import login
login("hf_YOUR_TOKEN_HERE")

# Cell 3 — Paste Stage 2's URL here, and set ngrok token for Instance 1
import os
os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_1"
os.environ["STAGE2_URL"]  = "https://aaa111.ngrok-free.app"  # ← paste Stage 2's URL here

# Cell 4 — Patch STAGE2_URL into the script before running
with open("stage1_server.py") as f:
    src = f.read()
src = src.replace(
    '"https://PASTE-STAGE2-NGROK-URL-HERE.ngrok-free.app"',
    f'"{os.environ["STAGE2_URL"]}"'
)
with open("stage1_server_patched.py", "w") as f:
    f.write(src)

exec(open("stage1_server_patched.py").read())
```

**Stage 1 will print:**
```
============================================================
[Stage 1] PUBLIC URL: https://bbb222.ngrok-free.app
  → Copy this URL into stage0_server.py as STAGE1_URL
  → e.g.  STAGE1_URL = "https://bbb222.ngrok-free.app"
============================================================
```

**Copy that URL. You will paste it into Stage 0 next.**

---

### Stage 0 (Kaggle Instance 0) — Boot Third

```python
# Cell 1 — Install
!pip install -q transformers accelerate flask pyngrok torch sentencepiece

# Cell 2 — HuggingFace auth
from huggingface_hub import login
login("hf_YOUR_TOKEN_HERE")

# Cell 3 — Paste Stage 1's URL here, and set ngrok token for Instance 0
import os
os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_0"
os.environ["STAGE1_URL"]  = "https://bbb222.ngrok-free.app"  # ← paste Stage 1's URL here

# Cell 4 — Patch STAGE1_URL into the script before running
with open("stage0_server.py") as f:
    src = f.read()
src = src.replace(
    '"https://PASTE-STAGE1-NGROK-URL-HERE.ngrok-free.app"',
    f'"{os.environ["STAGE1_URL"]}"'
)
with open("stage0_server_patched.py", "w") as f:
    f.write(src)

exec(open("stage0_server_patched.py").read())
```

**Stage 0 will print:**
```
============================================================
[Stage 0] PUBLIC URL: https://ccc333.ngrok-free.app
  → Share this with your client (client.py)
  → POST to https://ccc333.ngrok-free.app/generate
============================================================
```

---

### Client (your laptop) — Run Last

```bash
# Install
pip install requests

# Single prompt
python client.py --url https://ccc333.ngrok-free.app \
                 --prompt "Explain attention mechanisms"

# Interactive chat
python client.py --url https://ccc333.ngrok-free.app --interactive
```

---

## VRAM Budget (T4 = 15 GB each)

| | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| Weights | ~5.5 GB | ~5.0 GB | ~5.5 GB |
| KV cache | ~1.0 GB | ~1.0 GB | ~1.0 GB |
| Overhead | ~0.5 GB | ~0.5 GB | ~0.5 GB |
| **Total** | **~7.0 GB** ✅ | **~6.5 GB** ✅ | **~7.0 GB** ✅ |

All three instances fit comfortably within the T4's 15 GB with ~8 GB headroom each —
more breathing room than the 2-stage split.

---

## Token Transfer Size Per Step

Hidden size for Llama 3 8B = 4096 dimensions.
Per token sent: `1 × 1 × 4096 × 2 bytes (fp16)` = **8 KB per token** — negligible.
This is the same payload size as the 2-stage pipeline; splitting into 3 stages does not
increase transfer size, only the number of hops.

---

## Latency Breakdown (estimated, T4 + ngrok)

| Step | Time |
|---|---|
| Stage 0 compute (per token) | ~8–14 ms |
| ngrok hop S0 → S1 (8 KB) | ~10–30 ms |
| Stage 1 compute (per token) | ~8–14 ms |
| ngrok hop S1 → S2 (8 KB) | ~10–30 ms |
| Stage 2 compute (per token) | ~8–14 ms |
| **Total per token** | **~44–102 ms** |
| **Effective throughput** | **~10–23 tok/s** |

The extra ngrok hop (vs 2-stage) costs ~10–30 ms per token. Use Cloudflare Tunnels
to cut each hop to ~2–5 ms if latency matters.

---

## Verifying All Three Stages Are Healthy

Before running the client, confirm each stage is reachable:

```bash
# From your laptop — replace URLs with your actual ngrok URLs
curl https://aaa111.ngrok-free.app/health   # Stage 2
curl https://bbb222.ngrok-free.app/health   # Stage 1
curl https://ccc333.ngrok-free.app/health   # Stage 0
```

Each should return something like:
```json
{"stage": 2, "status": "ok", "vram_gb": 7.01}
```

---

## Common Issues

**`CUDA out of memory`**
- Verify you selected GPU in Kaggle (Settings → Accelerator → GPU T4 x1)
- Ensure no other processes are using VRAM: `!nvidia-smi`
- Each instance only loads its own slice of layers — if OOM occurs, check that the
  full model wasn't accidentally loaded with `device_map="auto"`

**`ConnectionError` to Stage 1 or Stage 2**
- Stages must be booted in order — Stage 2 fully loaded before Stage 1 starts,
  Stage 1 fully loaded before Stage 0 starts
- Check ngrok URLs were copied correctly (no trailing slash, no extra spaces)
- ngrok free tier: 1 simultaneous tunnel per account — confirm you used 3 separate accounts

**`STAGE2_URL not set` error on Stage 1**
- The patching step in Cell 4 must run before `exec()` — check the replace string
  matches exactly: `"https://PASTE-STAGE2-NGROK-URL-HERE.ngrok-free.app"`

**`401 Unauthorized` from HuggingFace**
- Request access to `meta-llama/Meta-Llama-3-8B` at huggingface.co/meta-llama
- Token must have `read` permission
- All three instances need the same HF token (or separate valid tokens)

**Slow generation**
- ngrok free tier adds latency; Cloudflare Tunnel (`cloudflared tunnel`) is faster
  for sustained use and has no tunnel-per-account limit
- The 3-stage pipeline has 2 ngrok hops per token vs 1 in the 2-stage version —
  switching to Cloudflare Tunnels on all instances is the single biggest speedup

**Stage 1 returns wrong token / garbled output**
- Verify `SPLIT_LAYER_START = 11` and `SPLIT_LAYER_END = 22` in `stage1_server.py`
- Verify `SPLIT_LAYER = 11` in `stage0_server.py` and `SPLIT_LAYER = 22` in `stage2_server.py`
- Layer boundaries must be consistent: S0 owns 0–10, S1 owns 11–21, S2 owns 22–31

---

## What to Experiment With

1. **Change split points** — try (8, 16, 24) or (10, 20, 22) and measure latency + VRAM changes
2. **Tensor quantization** — quantize hidden states to int8 before each hop (reduces transfer 2×, adds ~1 ms encode/decode)
3. **Cloudflare Tunnels** — replace ngrok with `cloudflared tunnel --url localhost:500X` on each instance; no account limit and lower latency
4. **Batched generation** — send multiple prompts simultaneously, measure throughput scaling across 3 stages
5. **KV cache analysis** — log VRAM usage per step on each instance to observe cache growth independently
6. **Layer profiling** — time each stage's forward pass per token to identify the bottleneck stage
7. **4-instance split** — extend the relay pattern in `stage1_server.py` to add a fourth stage (layers ~8 each)
