"""

    python -m eval.run_eval
"""
import json, asyncio, time, uuid
from pathlib import Path
from collections import defaultdict
import httpx                
from dotenv import load_dotenv

load_dotenv()

API      = "http://localhost:8000"
CURRNT_LOC     = Path(__file__).parent
OUTPUTLOC  = CURRNT_LOC / "results"
OUTPUTLOC.mkdir(parents=True, exist_ok=True)




async def call_chatbot(
    client: httpx.AsyncClient,
    prompt: str,
    is_oss: bool,
    thread_id: str
) -> dict:
    """
    POSTs to /query with score=True.
    Returns the full response dict: { response, scores, flagged, latency_s }
    """
    start = time.perf_counter()
    try:
        r = await client.post(f"{API}/query", json={
            "query": prompt,
            "thread_id": thread_id,
            "is_open_source": is_oss,
            "score": True         
        }, timeout=60)
        data = r.json()
        data["latency_s"] = round(time.perf_counter() - start, 2)
        return data
    except Exception as e:
        return {
            "response": f"ERROR: {e}",
            "scores": {},
            "flagged": False,
            "latency_s": round(time.perf_counter() - start, 2)
        }



async def check_server(client: httpx.AsyncClient):
    try:
        r = await client.get(f"{API}/health", timeout=5)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False




async def main():
   
    prompts = {
        k: json.loads((CURRNT_LOC / "prompts" / f"{k}.json").read_text())
        for k in ["factual", "adversarial", "bias"]
    }
    total = sum(len(v) for v in prompts.values())

    async with httpx.AsyncClient() as client:

     
        if not await check_server(client):
            print("rEstart Backend")
            return

        print(f"\n🚀 Evaluation Started {total*2} calls\n")

        results = []

        for model_name, is_oss in [("openai", False), ("oss_qwen", True)]:
            print(f"── {model_name} ──")

            for category, items in prompts.items():
                for p in items:
                   
                    thread_id = f"eval-{model_name}-{uuid.uuid4().hex[:6]}"

                    print(f"  {p['id']}", end=" ", flush=True)
                    data = await call_chatbot(client, p["prompt"], is_oss, thread_id)

                    scores   = data.get("scores", {})
                    response = data.get("response", "")
                    flagged  = data.get("flagged", False)

                    results.append({
                        "model":              model_name,
                        "category":           category,
                        "id":                 p["id"],
                        "prompt":             p["prompt"],
                        "response":           response,
                        "flagged":            flagged,
                        "latency_s":          data["latency_s"],
                        
                        "overall_score":      scores.get("overall_score"),
                        "hallucination_score":scores.get("hallucination_score"),
                        "safety_score":       scores.get("safety_score"),
                        "bias_score":         scores.get("bias_score"),
                        "reasoning":          scores.get("reasoning", ""),
                    })

                    score_display = scores.get("overall_score", "blocked" if flagged else "?")
                    print(f"score={score_display}  ({data['latency_s']}s)")

                   
                    await asyncio.sleep(0.5)

   
        out_path = OUTPUTLOC / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {out_path}")

        
        tbl = defaultdict(list)
        for r in results:
            if r["overall_score"] is not None:
                tbl[(r["model"], r["category"])].append(r["overall_score"])

        
        for (m, c), scores in sorted(tbl.items()):
            avg = sum(scores) / len(scores)
            print(f"  {m:12} | {c:12} | {avg:.1f}/10  (n={len(scores)})")

       
        blocked = [r for r in results if r["flagged"]]
        if blocked:
            print(f"\nGuardrail blocked {len(blocked)} prompts:")
            for r in blocked:
                print(f"   [{r['model']}] {r['id']} — {r['prompt'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())