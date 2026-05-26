# from fastapi import FastAPI
# from dotenv import load_dotenv
# from models.chat_model import ChatModel
# from assistants.openai_assistant import generate_response 

# import uvicorn


# app = FastAPI(
#     title="Eval Agents",
#     debug= True
# )



# SYSTEM_PROMPT = {
#     "role": "system",
#     "content": "You are a helpful AI assistant."
# }


# messages = []
# sessions = {}

# messages.append(SYSTEM_PROMPT)

# @app.post("/query")
# async def query_agent(request: ChatModel):

#     session_id = request.thread_id
#     if session_id not in sessions:
#         sessions[session_id] = [SYSTEM_PROMPT]

#     messages = sessions[session_id]
    
#     messages.append({
#            "role": "user",
#         "content": request.query
#     })
#     messages = messages[-20:]
#     response = await generate_response(
#         messages=messages,
#         is_open_source=False
#     )
    

    
#     messages = sessions[session_id]
#     messages.append({
#            "role": "assistant",
#         "content": response
#     }
#     )
    
#     print(sessions)
#     return {
#         "response": response,
#         "session_id": session_id
#     }


# if __name__ == "__main__":
#     uvicorn.run(app)



import os


from fastapi import FastAPI, Request
from dotenv import load_dotenv
from models.chat_model import ChatModel
from assistants.openai_assistant import generate_response
import uvicorn
from langfuse.openai import AsyncOpenAI
from contextlib import asynccontextmanager
from langfuse import get_client
import re
load_dotenv()

BLOCKED = [
    r"ignore (all |your )?(previous |prior )?instructions",
    r"you are (now )?(dan|jailbreak|evil ai)",
    r"(make|build|synthesize).{0,30}(bomb|explosive|meth)",
    r"(write|create).{0,20}(malware|ransomware|virus)",
    r"(forget|bypass|disregard).{0,20}(rules|guidelines|restrictions)",
]
_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED]
 
async def is_safe(text: str, openai_client: AsyncOpenAI) -> tuple[bool, str]:
    for pattern in _BLOCKED_RE:
        if pattern.search(text):
            return False, "blocklist"
    try:
        result = await openai_client.moderations.create(input=text)
        if result.results[0].flagged:
            return False, "moderation_api"
    except Exception:
        pass  # don't block if moderation API fails
    return True, "ok"

@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Starting up...")
    app.state.openai_client =AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    yield
    print("Shutting down...")

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful AI assistant."}

# session_id -> list of messages
sessions: dict[str, list] = {}
app = FastAPI(title="Eval Agents",lifespan = lifespan)

@app.post("/query")
async def query_agent(request: ChatModel, req:Request):
    safe, reason = await is_safe(request.query,req.app.state.openai_client)
    if not safe:
        return {
            "response" : "I'm not able to help with that request",
            "flag_reason" : reason,
            "session_id" : request.thread_id,
            "flagged" : True
        }
    
    session_id = request.thread_id
    langfuse_client = get_client()
    trace_id = langfuse_client.create_trace_id()
    
    if session_id not in sessions:
        sessions[session_id] = [SYSTEM_PROMPT]

    
    sessions[session_id].append({"role": "user", "content": request.query})

    history = [SYSTEM_PROMPT] + sessions[session_id][1:][-20:]

    

    response = await generate_response(
        messages=history,
        is_open_source=request.is_open_source,
        req=req,
        session_id=session_id,
         langfuse_observation_id=trace_id
    )

    # Save assistant reply to session
    sessions[session_id].append({"role": "assistant", "content": response})

    return {
        "response": response,
        "session_id": session_id,
        "model": "oss" if request.is_open_source else "openai"
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, )
