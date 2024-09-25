from datasets import load_dataset
from datetime import date, time, datetime
import dataclasses
from typing import List, Optional
from openai import OpenAI
import os
import yaml

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))

# resp = client.chat.completions.create(
#     messages = [{ 
#         "role": "user", 
#         "content": "Write short complaint to The Boston Globe about the rat problem at Northeastern CS. Blame the math department. No more than 4 sentences." 
#     }],
#     model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
#     temperature=0)
# print(resp.choices[0].message.content)

@dataclasses.dataclass
class Flight:
    id: int
    date: date
    airline: str
    flight_number: str
    origin: str
    destination: str
    departure_time: time
    arrival_time: time
    available_seats: int


def parse_flight(flight):
    return Flight(
        id=flight["id"],
        date=datetime.strptime(flight["date"], "%Y-%m-%d").date(),
        airline=flight["airline"],
        flight_number=flight["flight_number"],
        origin=flight["origin"],
        destination=flight["destination"],
        departure_time=datetime.strptime(flight["departure_time"], "%H:%M").time(),
        arrival_time=datetime.strptime(flight["arrival_time"], "%H:%M").time(),
        available_seats=flight["available_seats"],
    )


def load_flights_dataset() -> List[Flight]:
    return [
        parse_flight(flight)
        for flight in load_dataset("nuprl/llm-systems-flights", split="train")
    ] 

@dataclasses.dataclass
class AgentResponse:
    """
    The superclass for all agent responses.
    """
    text: str

@dataclasses.dataclass
class FindFlightsResponse(AgentResponse):
    """
    The agent used the `find_flights` tool and found the following flights.
    """
    available_flights: List[int]


@dataclasses.dataclass
class BookFlightResponse(AgentResponse):
    """
    The agent used the `book_flight` tool and booked the following flight.
    """
    booked_flight: Optional[int]


@dataclasses.dataclass
class TextResponse(AgentResponse):
    pass

SYSTEM_PROMPT = """
You are a helpful travel agent. Respond to queries with code that uses
the already defined following functions:

def find_flights(origin: str, destination: str, date: datetime.date) -> list:
    # Returns a list of flight IDs that match the origin (represented by an airport code), destination (represented by an airport code), and date.
    ...

def book_flight(flight_id: int) -> Optional[int]:
    # Books a flight with the given ID and returns the booking ID.
    ...

Return the result in a variable called result.

Today's date is September 1 2024.
"""

class Agent:

    # The complete conversation with the LLM, including the system prompt.
    conversation: List[dict]
    # The formatted response from the last tool call.
    text_prefix: Optional[str]
    # The current database of flights. The tools update this database.
    flights: List[Flight]
    client: OpenAI
    # Global variables used in tool calls.
    program_state: dict

    def find_flights(self, origin: str, destination: str, date: date) -> List[Flight]:
        pass
        
    def book_flight(self, flight_id: int) -> Optional[int]:
        pass

    def say(self, user_message: str) -> AgentResponse:
        pass


class EvaluationResult:
    """
    The result of evaluating an agent on a benchmark.
    """

    # The score of the agent on the benchmark, between 0.0 and 1.0.
    score: float
    # The conversation with the agent.
    conversation: List[dict]

def eval_agent(client: OpenAI, benchmark_file: str, flights: List[Flight]) -> float:
    """
    Evaluate the agent on the given benchmark YAML file.
    """
    agent = Agent(flights=flights, client=client)
    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)
    for n, step in enumerate(steps):
        response = agent.say(step["prompt"])
        match step["expected_type"]:
            case "text":
                if not isinstance(response, TextResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "find-flights":
                if not isinstance(response, FindFlightsResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                if set(response.available_flights) != set(step["expected_result"]):
                    return EvaluationResult(n / len(steps), agent.conversation)
            case "book-flight":
                if not isinstance(response, BookFlightResponse):
                    return EvaluationResult(n / len(steps), agent.conversation)
                if response.booked_flight != step["expected_result"]:
                    return EvaluationResult(n / len(steps), agent.conversation)
    return EvaluationResult(1.0, agent.conversation)  


eval_agent(client, "benchmark.yaml", load_flights_dataset())