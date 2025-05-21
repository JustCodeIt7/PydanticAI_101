"""
structured_city_info.py  –  PydanticAI Masterclass · Video 5
------------------------------------------------------------
Demonstrates *structured* LLM output validated by Pydantic.

• Defines a `CityInfo` Pydantic model describing the expected JSON shape
• Instantiates an Agent with `output_type=CityInfo`
• Sends a question and receives a fully‑typed Python object

Prerequisite:
  • Set OPENAI_API_KEY (or switch to another provider / TestModel)
Run:
  python structured_city_info.py
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent

# ── 1. Define desired response schema ──────────────────────────────────────
class CityInfo(BaseModel):
    city_name: str = Field(description="The name of the city")
    country: str = Field(description="The country the city is in")
    population: int = Field(description="Estimated population")
    landmarks: List[str] = Field(description="List of famous landmarks")

# ── 2. Configure an Agent that MUST return CityInfo objects ────────────────
agent = Agent(
    "openai:gpt-4o",                              # swap provider if desired
    system_prompt="Provide factual data about the city in JSON.",
    output_type=CityInfo,                         # ← critical line
)

# ── 3. Ask for a city – the response will be type‑validated ────────────────
result = agent.run_sync("Tell me about Kyoto, Japan.")

# `result.output` is already a CityInfo instance (not a raw string!)
city: CityInfo = result.output

# ── 4. Use structured data safely in Python code ───────────────────────────
print(f"\n📍 {city.city_name}, {city.country}")
print(f"   Population (est.): {city.population:,}")
print("   Top landmarks:")
for lm in city.landmarks:
    print("    •", lm)

# You can also serialize / persist the object:
# json_data = city.model_dump_json(indent=2)
# open("kyoto_info.json", "w").write(json_data)
