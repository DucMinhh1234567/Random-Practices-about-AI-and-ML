import re
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_HOST = "https://research.neu.edu.vn/ollama"
OLLAMA_MODEL = "qwen3:8b"


def ollama_chat(messages: list[dict], max_tokens: int = 10000) -> str:
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

    def run(self, task):
        print(f"{self.name} is running")

        return ollama_chat(
            messages=[
                {"role": "system", "content": self.role.strip()},
                {"role": "user", "content": task},
            ],
        )


research_agent = Agent(
    "Research Agent",
    """
    You are an expert travel researcher.
    Your job:
    - Find popular attractions
    - Find hidden gems
    - Suggest local experiences
    - Recommend best places
    """
)

activity_agent = Agent(
    "Activity Planner Agent",
    """
    You are a professional travel planner.
    Your job:
    - Create daily activities
    - Plan sightseeing
    - Recommend food experiences
    - Organize activities logically
    """
)

budget_agent = Agent(
    "Budget Agent",
    """
    You are a travel budget expert.
    Calculate:
    - Estimated flight cost
    - Visa requirements
    - Visa fees if needed
    - Hotel cost
    - Food expenses
    - Transport cost
    - Activity costs
    Create an approximate total trip budget.
    Keep response short.
    """
)

final_agent = Agent(
    "Final Travel Assistant",
    """
    You are a professional travel planner.
    Create the final itinerary.
    Include:
    1. Short trip overview
    2. Visa information
    3. Estimated flight cost
    4. Day-wise plan
    5. Food suggestions
    6. Total estimated budget
    Keep everything under 700 words.
    """
)


# User input
starting_location = input(
    "Where are you flying from? "
)

destination = input(
    "Where do you want to travel? "
)

days = input(
    "How many days is your trip? "
)

travelers = input(
    "How many travelers? "
)

budget = input(
    "What is your budget? (low/medium/high) "
)


interests = input(
    "What are your interests? "
)

# Create request for AI agents

user_request = f"""

Create a travel plan with these details:

Flying From:
{starting_location}

Destination:
{destination}

Trip Duration:
{days} days

Number of Travelers:
{travelers}

Budget Level:
{budget}

Interests:
{interests}

Include:
- Visa requirements
- Estimated flight cost
- Places to visit
- Activities
- Food recommendations
- Total estimated budget

"""

print("\nCreating your AI travel plan...\n")


#Multi-Agent Workflow

#Agent 1 researches destination
research = research_agent.run(
    user_request
)
print("\n--- Research Completed ---")

#Agent 2 creates activities
activities = activity_agent.run(
    research
)
print("\n--- Activities Planned ---")

#Agent 3 calculates budget
budget = budget_agent.run(
    activities
)
print("\n--- Budget Created ---")

#Agent 4 creates final itinerary
final_plan = final_agent.run(
    f"""
    Research:
    {research}

    Activities:
    {activities}

    Budget:
    {budget}

    Create final travel plan.
    """
)

slug = re.sub(r"[^\w]+", "_", destination.lower()).strip("_")[:40]
output_path = Path(__file__).parent / f"travel_plan_{slug}_{datetime.now():%Y%m%d_%H%M%S}.md"

markdown = f"""# Travel Plan: {destination}

- **From:** {starting_location}
- **Duration:** {days} days
- **Travelers:** {travelers}
- **Interests:** {interests}

---

{final_plan}
"""

output_path.write_text(markdown, encoding="utf-8")
print(f"\nTravel plan saved to: {output_path}")