"""
╔══════════════════════════════════════════════════════════╗
║         LLAMA 3 8B PIPELINE — CLIENT                    ║
║         Run this from your laptop or any machine        ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python client.py --url https://YOUR-STAGE0-NGROK.ngrok-free.app
  python client.py --url https://... --prompt "Explain transformers"
  python client.py --url https://... --interactive
"""

import argparse, requests, json, time, sys, uuid

def stream_generate(base_url: str, prompt: str,
                    max_new_tokens: int = 200,
                    temperature: float = 0.8,
                    top_p: float = 0.95,
                    session_id: str = None) -> str:
    """
    POST to Stage 0 /generate, stream tokens back, print in real time.
    Returns the full generated text.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    payload = {
        "prompt":         prompt,
        "max_new_tokens": max_new_tokens,
        "temperature":    temperature,
        "top_p":          top_p,
        "session_id":     session_id,
    }

    print(f"\n[Prompt]: {prompt}")
    print(f"[Session: {session_id}]\n")
    print("─" * 60)
    sys.stdout.write("[Model]: ")
    sys.stdout.flush()

    full_text = ""
    t_start   = time.time()
    token_count = 0

    with requests.post(f"{base_url}/generate", json=payload, stream=True,
                       timeout=300) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_text   += chunk
                token_count += 1   # approximate; each chunk ≈ 1 token

    elapsed = time.time() - t_start
    tps     = token_count / elapsed if elapsed > 0 else 0
    print(f"\n{'─' * 60}")
    print(f"[Stats] {token_count} tokens in {elapsed:.1f}s "
          f"({tps:.1f} tok/s)\n")

    return full_text


def health_check(base_url: str):
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        d = r.json()
        print(f"✅ Stage 0 healthy | VRAM: {d.get('vram_gb', '?'):.2f} GB")
    except Exception as e:
        print(f"❌ Stage 0 unreachable: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Llama 3 Pipeline Client")
    parser.add_argument("--url",           required=True,
                        help="Stage 0 ngrok URL (e.g. https://abc123.ngrok-free.app)")
    parser.add_argument("--prompt",        default="Explain how transformers work in 3 sentences.")
    parser.add_argument("--max_new_tokens", type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float, default=0.95)
    parser.add_argument("--interactive",    action="store_true",
                        help="Enter interactive chat mode")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    health_check(base_url)

    if args.interactive:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n🦙 Llama 3 8B Pipeline — Interactive Mode")
        print(f"   Session: {session_id}")
        print(f"   Type 'quit' to exit, 'new' to start fresh session\n")
        conversation = ""

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "new":
                session_id   = str(uuid.uuid4())[:8]
                conversation = ""
                print(f"[New session: {session_id}]")
                continue
            if not user_input:
                continue

            # Simple chat formatting
            conversation += f"\n\nUser: {user_input}\nAssistant:"
            response = stream_generate(
                base_url,
                prompt=conversation,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                session_id=session_id,
            )
            conversation += " " + response.strip()
    else:
        stream_generate(
            base_url,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
