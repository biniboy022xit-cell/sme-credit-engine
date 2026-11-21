"""LLM agent layer: explains credit assessments using real tool calls.

Design choices, deliberate and important:
- The MODEL and RULES decide the risk score, band and warnings (src/scoring.py,
  src/rules.py). The LLM never invents or overrides these numbers.
- The agent's job is narrow: call tools to fetch real data, then summarise it
  in plain language for a human credit officer. It makes no approve/decline
  decision — that line is drawn deliberately, see suggested_action() in
  src/scoring.py.
- Runs against a free hosted API (Groq), not a local model. A 1-3B local
  model on a CPU-only laptop (no NVIDIA GPU) would be too slow for an
  interactive demo and adds nothing the API doesn't already do better.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from assets import list_assets, total_collateral
from db_scoring import get_business, score_and_save

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are an assistant that helps a credit officer review SME loan
applications. You do not make approval or rejection decisions yourself.

Use the tools to get the business's real risk score, warnings and registered
assets. Never invent numbers. All monetary amounts are in Zambian Kwacha (ZMW). Summarise what the tools return in plain,
concise language, and suggest what a human officer should check next. Always
state the exact risk band and probability the tools returned. If a tool
returns an error (such as an unknown business ID), say so plainly instead of
guessing.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_risk_assessment",
            "description": "Run the credit risk model on a business and save the result. "
                            "Returns probability of default, risk band, suggested action, "
                            "top factors, and any early-warning flags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "business_id": {"type": "string", "description": "e.g. SME00001"},
                },
                "required": ["business_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_collateral",
            "description": "List the registered collateral assets for a business and "
                            "their total value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "business_id": {"type": "string", "description": "e.g. SME00001"},
                },
                "required": ["business_id"],
            },
        },
    },
]


def _call_tool(name: str, args: dict) -> dict:
    """Execute a tool call and return a JSON-serialisable result. Never raises."""
    try:
        if name == "get_risk_assessment":
            raw = get_business(args["business_id"])  # raises ValueError if unknown
            result = score_and_save(args["business_id"])
            return {
                "business_name": raw["business_name"],
                "probability_of_default": result["probability_of_default"],
                "risk_band": result["risk_band"],
                "suggested_action": result["suggested_action"],
                "raising_risk": result["raising_risk"],
                "lowering_risk": result["lowering_risk"],
                "warnings": [w["message"] for w in result["warnings"]],
            }
        if name == "get_collateral":
            df = list_assets(args["business_id"])
            return {
                "assets": df[["description", "category", "value"]].to_dict("records")
                if not df.empty else [],
                "total_value": total_collateral(args["business_id"]),
            }
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:  # noqa: BLE001 - tool errors must reach the model as data, not crash
        return {"error": str(e)}


def run_agent(question: str, max_steps: int = 4) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = _call_tool(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Stopped after too many tool-call steps without a final answer."


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Assess SME00001 and tell me if I should be worried."
    print(run_agent(q))
