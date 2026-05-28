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
                    "expression": {"type": "string", "description": "A safe math expression e.g. '100 / 4'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for current information. Use when the user asks about "
                "recent news, live data, current events, or anything after your training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]