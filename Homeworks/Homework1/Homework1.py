from openai import OpenAI

client = OpenAI(base_url=URL, api_key=KEY)

resp = client.completions.create(
    model="Llama3.1-8B-Base",
    temperature=0.2,
    max_tokens=100,
    stop=["\n"],
    prompt="Shakespeare was"

)
print(resp.choices[0].text)

## Zero-Shot Prompting

# Task 1
from typing import List, Optional

def prompt_zero_shot(problem: str) -> str:
    pass

def extract_zero_shot(completion: str) -> Optional[int]:
    pass

# Task 2
def accuracy_zero_shot(problems: List[dict]) -> float:
    pass

# Task 3
import datasets

TEST = datasets.load_dataset("nuprl/llm-systems-math-word-problems", split="test")

print(accuracy_zero_shot(TEST))

## Few-Shot Prompting
# Task 4
def prompt_few_shot(problem: str) -> str:
    pass

def extract_few_shot(completion: str) -> Optional[int]:
    pass

def solve_few_shot(problem: str) -> Optional[int]:
    pass

def accuracy_few_shot(problems: List[dict]) -> float:
    pass

## Chain-of-Thought Prompting
# Task 5
def prompt_cot(problem: str) -> str:
    pass

def extract_few_shot(completion: str) -> Optional[int]:
    pass

def solve_few_shot(problem: str) -> Optional[int]:
    pass

def accuracy_few_shot(problems: List[dict]) -> float:
    pass

## Program-Aided Language Models
# Task 6
def prompt_pal(problem: str) -> str:
    pass

def extract_pal(completion: str) -> Optional[int]:
    pass

def solve_pal(problem: str) -> Optional[int]:
    pass

def accuracy_pal(problems: List[dict]) -> float:
    pass