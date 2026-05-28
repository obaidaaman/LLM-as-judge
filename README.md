AI Assistant Evaluation Project
Compare OpenAI GPT-4.1 (frontier) vs Qwen 2.5 (open source) on hallucination, safety, and bias.
Stack

Backend: FastAPI + session memory
Frontend: Streamlit (side-by-side chat)
Observability: Langfuse (traces every call)
Guardrails: Regex blocklist + OpenAI Moderation API
Tool use: get_current_date, calculate (OpenAI model only)
OSS deployment: Hugging Face Spaces (Qwen2.5-0.5B)
Eval: LLM-as-judge (GPT-4o scores both models)


Tradeoffs & Improvements

Sessions lost on restart → Redis for production
Tool use not available on OSS path → Use of an agent framework like LangGraph I will prefer
Single judge model → Ensemble judging reduces bias
HF Inference API has rate limits → Modal/Replicate for guaranteed SLA



