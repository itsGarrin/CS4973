from datasets import load_dataset
from datetime import date, time, datetime
import dataclasses
from typing import List, Optional
from openai import OpenAI
import os
import yaml
from pathlib import Path

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
Today's date is September 1 2022. You are a helpful travel agent that has access to tools to perform actions. Respond to queries with a single code block that uses
the already defined following functions (tools). For every prompt you recieve, you will decide which available tool to use from 
the following list if the response makes sense. Don't respond with a result just to provide a response. Only respond with a result if it makes sense to do so.

- `find_flights(origin: str, destination: str, date: datetime.date) -> list` - Returns a list of Flight objects that match the origin (represented by an airport code), destination (represented by an airport code), and date.
- `book_flight(flight_id: int) -> Optional[int]` - Books a flight with the given ID. Returns the flight ID if the booking was successful, or `None` if the flight was not found.

The flight object has an attribute `available_seats` that represents the number of available seats on the flight.
It also has an attribute `airline` that represents the airline of the flight.

Here are some example prompts and the tools you should use:

"Find me a flight from BOS to LAX on February 1, 2023."
- Tool: `find_flights("BOS", "LAX", datetime.date(2023, 2, 1))`

"Book the flight with an ID of 123."
- Tool: `book_flight(flight_id=123)`

Return back the result in a variable called `result`. It should be in the following format:
['find-flights', [123, 456, 789]]  # If you are using the find_flights function
['book-flight', 123]  # If you are using the book_flight function
"""

class Agent:

    def __init__(self, conversation: List[dict], flights: List[Flight], client: OpenAI):
        self.conversation = conversation
        self.flights = flights
        self.client = client

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
        flights = []

        for flight in self.flights:
            if flight.origin == origin and flight.destination == destination and flight.date == date:
                flights.append(flight)

        return flights
        
    def book_flight(self, flight_id: int) -> Optional[int]:
        for flight in self.flights:
            if flight.id == flight_id and flight.available_seats > 0:
                return flight.id

        return None
    
    def extract_code(self, resp_text):
        code_start = resp_text.find("```")
        code_end = resp_text.rfind("```")
        if code_start == -1 or code_end == -1:
            return "pass"
        return resp_text[code_start + 3 + 7:code_end]

    def say(self, user_message: str) -> AgentResponse:
        # print(user_message, "\n\n")
        self.conversation.append({"role": "user", "content": user_message})
        globals = { "find_flights": self.find_flights, "book_flight": self.book_flight, "Flight": Flight, "result": None, "datetime": datetime }

        # if number of tokens in conversation greater than 2048, print
        if len("".join([x["content"] for x in self.conversation])) > 5000:
            print(self.conversation)
            return TextResponse(text="Conversation too long. Please start a new conversation.")

        resp = self.client.chat.completions.create(
        messages = self.conversation,
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0)
        resp_text = resp.choices[0].message.content
        print(resp_text)

        exec(self.extract_code(resp_text), globals)

        self.conversation.append({"role": "system", "content": resp_text})

        # print(globals["result"])
        if globals["result"] is None:
            return TextResponse(text=resp_text)
        if globals["result"][0] == "find-flights":
            # flight_ids = [flight.id for flight in globals["result"][1]]
            return FindFlightsResponse(text=resp_text, available_flights=globals["result"][1])
        elif globals["result"][0] == "book-flight":
            return BookFlightResponse(text=resp_text, booked_flight=globals["result"][1])
        else:
            return TextResponse(text=resp_text)



class EvaluationResult:
    """
    The result of evaluating an agent on a benchmark.
    """

    def __init__(self, score: float, conversation: List[dict]):
        self.score = score
        self.conversation = conversation

    # The score of the agent on the benchmark, between 0.0 and 1.0.
    score: float
    # The conversation with the agent.
    conversation: List[dict]

CONVO = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I need to go from ATL to SEA on Jan 9 or Jan 11"}, 
    {"role": "system", "content": '''```python
from datetime import date
flights = []
flights += find_flights("ATL", "SEA", date(2023, 1, 9)) # find flights on the first date
flights += find_flights("ATL", "SEA", date(2023, 1, 11)) # find flights on the second date
flights = [flight.id for flight in flights]
result = ['find-flights', flights]
print(result)
    ```'''},
    {"role": "user", "content": "Book the earliest flight for the journey"},
    {"role": "system", "content": '''```python
from datetime import date
flights = find_flights("ATL", "SEA", date(2023, 1, 9))
flights = [flight.id for flight in flights]
result = ['book-flight', book_flight(flights[0])] # book the earliest flight
print(result)
    ```'''},
]

def eval_agent(client: OpenAI, benchmark_file: str, flights: List[Flight]) -> EvaluationResult:
    """
    Evaluate the agent on the given benchmark YAML file.
    """
    agent = Agent(flights=flights, client=client, conversation=CONVO.copy())
    with open(benchmark_file, "r") as file:
        steps = yaml.safe_load(file)
    for n, step in enumerate(steps):
        response = agent.say(step["prompt"])
        # print(response)
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
