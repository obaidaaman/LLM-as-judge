import os
# from openai import AsyncOpenAI
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

from langfuse import observe, propagate_attributes, get_client
from fastapi import Request
import json
load_dotenv()

# openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
hf_client     = AsyncInferenceClient(token=os.getenv("HF_TOKEN"))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns today's date. Use when the user asks about the current date or time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a simple math expression like '12 * 4 + 3'. Use for arithmetic questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A safe math expression, e.g. '100 / 4'"}
                },
                "required": ["expression"]
            }
        }
    }
]

def run_tool(name: str, args: dict) -> str:
   
    if name == "get_current_date":
        from datetime import date
        return str(date.today())
 
    if name == "calculate":
        try:
    
            result = eval(args["expression"], {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
 
    return "Unknown tool"

@observe()
async def generate_response(messages: list,
                            req: Request,
                            session_id:str, 
                            is_open_source: bool = False, 
                            **kwargs) -> str:
   

   with propagate_attributes(
      session_id=session_id,
      tags=["oss" if is_open_source else "openai"],
      metadata={"env" : "development", "model_type": "oss" if is_open_source else "frontier"}
   ):
              
       if is_open_source:
            try:
                
                
                res = await hf_client.chat_completion(
                    model=os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                messages=messages,
                max_tokens=512
                )
                output = res.choices[0].message.content
                return output
            except Exception as e:
                raise
        # res = await hf_client.chat_completion(         
        #     model="Qwen/Qwen2.5-7B-Instruct",
        #     messages=messages,
        #     max_tokens=512
        # )
        # return res.choices[0].message.content
            

       res = await req.app.state.openai_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=messages,
        tools= TOOLS,
        tool_choice= "auto"

    )
       msg = res.choices[0].message
       if msg.tool_calls:
           tool_messages = list(messages) + [msg]
 
           for tc in msg.tool_calls:
                args   = json.loads(tc.function.arguments)
                result = run_tool(tc.function.name, args)
                tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
           final = await req.app.state.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=tool_messages
        )
           return final.choices[0].message.content
       
   return msg.content
      