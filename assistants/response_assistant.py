import os
# from openai import AsyncOpenAI
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
import httpx
from langfuse import observe, propagate_attributes
from fastapi import Request
import json
from utils.utils import TOOLS
load_dotenv()

hf_client     = AsyncInferenceClient(token=os.getenv("HF_TOKEN"))



async def tavily_search(query: str) -> str:
   
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: TAVILY_API_KEY not set. Using existing knowledge"
 
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",  
                    "max_results": 4,
                    "include_answer": True,     
                },
                timeout=30
            )
            data = resp.json()
            print(data)
 
        lines = []
 
        if data.get("answer"):
            lines.append(f"Summary: {data['answer']}\n")
 
        for r in data.get("results", [])[:4]:
            lines.append(f"- [{r.get('title','')}]({r.get('url','')})")
            if r.get("content"):
                lines.append(f"  {r['content'][:300]}...")
 
        return "\n".join(lines) if lines else "No results found."
 
    except Exception as e:
        return f"No search found for this query, We will stick to my knowledge base: {e}"
    


async def run_tool(name: str, args: dict) -> str:
 
    if name == "get_current_date":
        from datetime import date
        return str(date.today())
 
    if name == "calculate":
        try:
            result = eval(args["expression"], {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
 
    if name == "web_search":
        return await tavily_search(args["query"])
 
    return "Unknown tool"


async def handle_tool_calls(msg, messages: list, second_call_fn):

    if not getattr(msg, "tool_calls", None):
        return None

    tool_messages = list(messages)

    tool_messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in msg.tool_calls
        ]
    })

    for tc in msg.tool_calls:

        try:
            args = json.loads(tc.function.arguments)
        except Exception:
            args = {}

        result = await run_tool(
            tc.function.name,
            args
        )

        print("\nTOOL RESULT:\n")
        print(result)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result)
        })

    print("\nFINAL TOOL MESSAGES:\n")
    print(json.dumps(tool_messages, indent=2))

    final = await second_call_fn(tool_messages)

    return final.choices[0].message.content


@observe()
async def generate_response(
    messages: list,
    req: Request,
    session_id: str,
    is_open_source: bool = False,
    **kwargs
) -> str:
 
    with propagate_attributes(
        session_id=session_id,
        tags=["oss" if is_open_source else "openai"],
        metadata={"env": "development", "model_type": "oss" if is_open_source else "frontier"}
    ):
 
       
        if is_open_source:
            res = await hf_client.chat_completion(
                model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=512
            )
            msg = res.choices[0].message
 
         
            tool_result = await handle_tool_calls(
                msg, messages,
                second_call_fn=lambda msgs: hf_client.chat_completion(
                    model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                    messages=msgs,
                    max_tokens=512,
                    tools= TOOLS,
                )
            )
            return tool_result if tool_result is not None else msg.content
 
        
        client = req.app.state.openai_client
        res = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = res.choices[0].message
 
        tool_result = await handle_tool_calls(
            msg, messages,
            second_call_fn=lambda msgs: client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=msgs
            )
        )
        return tool_result if tool_result is not None else msg.content