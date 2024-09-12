import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))


## Zero-Shot Prompting

# Task 1
from typing import List, Optional


def prompt_zero_shot(problem: str) -> str:
    return problem + " Correct Answer: "


def extract_zero_shot(completion: str) -> Optional[int]:
    if completion.isdigit():
        return int(completion)


def solve_zero_shot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.1,
        prompt=prompt_zero_shot(problem),
        max_tokens=1
    )

    return extract_zero_shot(resp.choices[0].text)


# Task 2
def accuracy_zero_shot(problems: List[dict]) -> List[float]:
    accuracies = []

    for _ in tqdm(range(5)):
        total_correct = 0

        for item in problems:
            answer = solve_zero_shot(item["question"])

            if answer == item["answer"]:
                total_correct += 1
        accuracies.append(total_correct / len(problems))

    return accuracies


## Few-Shot Prompting
# Task 4
def prompt_few_shot(problem: str) -> str:
    return f"""
    Question: I ate 2 pears and 1 apple. How many fruit did I eat?
    Answer: 3

    Question: I had ten chocolates but ate one. How many remain?
    Answer: 9

    Question: {problem}
    Answer:"""


def extract_few_shot(completion: str) -> Optional[int]:
    # only extract digits from str
    return int("".join(filter(str.isdigit, completion)))


def solve_few_shot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B",
        temperature=0.2,
        prompt=prompt_few_shot(problem),
        max_tokens=100,
        stop=["\n"]
    )

    return extract_few_shot(resp.choices[0].text)


def accuracy_few_shot(problems: List[dict]) -> List[float]:
    accuracies = []

    for _ in tqdm(range(5)):
        total_correct = 0

        for item in problems:
            answer = solve_few_shot(item["question"])

            if answer == item["answer"]:
                total_correct += 1
        accuracies.append(total_correct / len(problems))

    return accuracies


## Chain-of-Thought Prompting
# Task 5
#def prompt_cot(problem: str) -> str:
#    pass


#def extract_few_shot(completion: str) -> Optional[int]:
#    pass


#def solve_few_shot(problem: str) -> Optional[int]:
#    pass


#def accuracy_few_shot(problems: List[dict]) -> float:
#    pass


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
