# AI Personal Assistant — Evaluation Project

Compare **OpenAI GPT-4.1** (frontier) vs **Qwen 2.5-7B** (open source) on factual accuracy, content safety, and bias.

---

## Demo

> Run locally — see setup below. Screenshots in `evaluation/charts/`.

---

## Project Structure

```
├── main.py                        # FastAPI backend — sessions, guardrails, judge
├── app.py                         # Streamlit chat UI — side-by-side comparison
├── assistants/
│   └── openai_assistant.py        # Both model callers + tool use + Langfuse tracing
├── models/
│   └── chat_model.py              # Request schema (query, thread_id, is_open_source, score)
├── evaluation/
│   ├── run_eval.py                # Runs all prompts through /query endpoint
│   ├── report.py                  # Generates comparison charts
│   └── prompts/
│       ├── factual.json           #  factual Q&A prompts
│       ├── adversarial.json       #  jailbreak/harmful prompts
│       └── bias.json              #  bias & fairness prompts

```

---

## Setup

### 1. Install dependencies

```bash
uv add -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:
```bash
uv venv
```

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-2025-04-14
OPENAI_JUDGE_MODEL=gpt-4.1-mini-2025-04-14
HF_TOKEN=hf_...
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
TAVILY_API_KEY=tv-....
# Langfuse — get from docker spin up
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Configure Docker

```bash
docker-compose up -d
```
 Register on the Langfuse portal on localhost:3000 after docker spin up
 Create Organisation and Project, Paste the credentials in .env


### 3. Run

```bash
# Terminal 1 — backend
uvicorn main:app --reload

# Terminal 2 — frontend
streamlit run app.py
```

### 4. Run evaluation

```bash
# Backend must be running first
python -m evaluation.eval_prompts      
python -m evaluation.report       
```

---

## Architecture Decisions

| Decision | Reason |
|---|---|
| Single `/query` endpoint with `is_open_source` flag | Keeps session memory, guardrails, and judge logic in one place — both models go through identical pipeline |
| In-memory session dict | Zero infrastructure for a demo/eval project. Trivial to swap for Redis in production |
| Two-layer guardrails (regex → moderation API) | Regex catches obvious attacks instantly (0ms), moderation API catches semantic attacks. Fast path first |
| LLM-as-judge scoring | Scalable evaluation without manual labelling. GPT-4o judge scores on 3 dimensions per response |
| `score: bool` flag on requests | Scoring adds ~1s latency. Off by default for normal chat, on for eval runs |
| Tool use on OpenAI only | HF Inference API does not reliably support function calling. OSS tool use would require a local deployment |
| Langfuse via wrapped client | Zero code change to instrument OpenAI calls. |

---

## Tradeoffs

- **In-memory sessions** → lost on server restart. Redis would solve this.
- **HF Inference API** → shared rate limits, cold starts up to 30s. Modal or HF Spaces with dedicated GPU would give consistent latency.
- **Single judge model (GPT-4o)** → may favour OpenAI-style responses. Combining judging (GPT-4o + Claude) would reduce bias.
- **25 eval prompts** → enough to demonstrate framework but not statistically rigorous. 200+ prompts per category would give confidence intervals.
- **No streaming** → responses block until complete. FastAPI SSE + Streamlit `st.write_stream` would improve UX for long answers.

---

## What I Would Improve With More Time

1. **Deploy OSS model to HF Spaces / Modal** — public URL, consistent latency, no cold starts
2. **Redis session store** — persistent memory across restarts, horizontal scaling
3. **Streaming responses** — better UX for long answers
4. **Larger eval set** — 50+ prompts per category with confidence intervals
5. **Ensemble judge** — multiple judge models to reduce single-model scoring bias
6. **Cost tracking** — log token usage per request, generate cost-per-1000-queries table
7. **Fine-grained Langfuse tagging** — tag by eval category, prompt ID, model for dashboard filtering
8. **CI eval pipeline** — run evals on every commit, alert on score regression

---

See `Evaluation_Report.pdf` for full charts and analysis.

---

## Bonus Features Implemented

- **Observability** — Langfuse traces every request (session ID, model type, latency, token count)
- **Guardrails** — Regex blocklist + OpenAI Moderation API on every input
- **Tool use** — `get_current_date`, `calculate` and `web search` tools available to OpenAI model/HF model
- **Live eval scores** — toggle in Streamlit UI scores each response in real time
- **Eval connected to chatbot** — `evaluation.py` calls `/query` endpoint, not models directly