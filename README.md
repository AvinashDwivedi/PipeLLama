# Llama 3 8B — 3-Instance Pipeline on Kaggle

```
Client → [stage0: layers 0–10] →(ngrok)→ [stage1: layers 11–21] →(ngrok)→ [stage2: layers 22–31 + head] → token
```

Three Kaggle T4 notebooks share the model. Hidden states (~8 KB) are passed between stages over ngrok.

---

## Prerequisites

- HuggingFace access to `meta-llama/Meta-Llama-3-8B` + a `read` token
- **3 separate ngrok accounts** (free tier = 1 tunnel per account) — get auth tokens from each dashboard

---

## Boot Order: Stage 2 → Stage 1 → Stage 0 → Client

### Stage 2 — Boot First
Open `stage2-cross-llm.ipynb`. In the first cell set your tokens:
```python
HF_TOKEN    = "hf_..."
NGROK_TOKEN = "..."   # ngrok account 2
```
Run all cells. Copy the printed ngrok URL (e.g. `https://aaa111.ngrok-free.app`).

### Stage 1 — Boot Second
Open `stage1-cross-llm.ipynb`. Set:
```python
HF_TOKEN    = "hf_..."
NGROK_TOKEN = "..."                              # ngrok account 1
STAGE2_URL  = "https://aaa111.ngrok-free.app"   # ← paste Stage 2 URL
```
Run all cells. Copy the printed ngrok URL.

### Stage 0 — Boot Third
Open `stage0-cross-llm.ipynb`. Set:
```python
HF_TOKEN    = "hf_..."
NGROK_TOKEN = "..."                              # ngrok account 0
STAGE1_URL  = "https://bbb222.ngrok-free.app"   # ← paste Stage 1 URL
```
Run all cells. Copy the printed ngrok URL — this is your client endpoint.

### Client — Run Last
```bash
pip install requests

# Single prompt
python client.py --url https://ccc333.ngrok-free.app --prompt "Hello!"

# Interactive chat
python client.py --url https://ccc333.ngrok-free.app --interactive
```

---

## Verify All Stages Are Up
```bash
curl https://aaa111.ngrok-free.app/health   # Stage 2
curl https://bbb222.ngrok-free.app/health   # Stage 1
curl https://ccc333.ngrok-free.app/health   # Stage 0
```
Each should return `{"stage": N, "status": "ok", "vram_gb": ...}`.

---

## Common Issues

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Confirm GPU enabled: Kaggle Settings → Accelerator → GPU T4 x1 |
| `ConnectionError` to next stage | Boot order wrong, or ngrok URL has a typo/trailing slash |
| `401 Unauthorized` from HuggingFace | Request model access at huggingface.co/meta-llama; token needs `read` permission |
| Garbled output | Check layer splits match: S0 owns 0–10, S1 owns 11–21, S2 owns 22–31 |
| Slow generation | Replace ngrok with Cloudflare Tunnels (`cloudflared tunnel --url localhost:500X`) |