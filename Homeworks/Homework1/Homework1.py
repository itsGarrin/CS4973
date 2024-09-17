import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))
model = "meta-llama/Meta-Llama-3.1-8B"


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
        model=model,
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
        model=model,
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
def prompt_cot(problem: str) -> str:
    return f"""
    Input: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?

    Reasoning: 7 pizzas are cut into 8 slices each. Thus the total number of slices is 7 * 8 = 56. Henry and his 3 friends want to share the pizza equally, so the slices are divided between 4 people. Each person gets 56 / 4 = 14 slices.

    Answer: 14

    Done

    Input: I have 5 boxes of pencils, each containing 12 pencils. I gave 8 pencils to my friend. How many pencils do I have left?

    Reasoning: I have 5 boxes of pencils, each containing 12 pencils. So, I have 5 * 12 = 60 pencils. I gave 8 pencils to my friend. Therefore, I have 60 - 8 = 52 pencils left.

    Answer: 52

    Done

    Input: A farmer has 3 fields. Each field has 20 rows of crops, and each row has 15 plants. If the farmer sells 100 plants, how many plants does he have left?

    Reasoning: A farmer has 3 fields. Each field has 20 rows of crops, and each row has 15 plants. So, the total number of plants is 3 * 20 * 15 = 900. If the farmer sells 100 plants, he has 900 - 100 = 800 plants left.

    Answer: 800

    Done

    Input: A school has 200 students. They are divided into 4 classes equally. If 10 students from each class join a sports event, how many students are left in the classes?

    Reasoning: A school has 200 students. They are divided into 4 classes equally. So, each class has 200 / 4 = 50 students. If 10 students from each class join a sports event, then 10 * 4 = 40 students join the event. Therefore, 200 - 40 = 160 students are left in the classes.

    Answer: 160

    Done

    Input: {problem}
    Reasoning:"""


def extract_cot(completion: str) -> Optional[float]:
    try:
        # Look for the answer by splitting at "Answer:" and extracting the part after that
        answer_part = completion.split("Answer:")[-1].strip()

        # Extract numeric value including decimals, ignoring non-numeric characters like $, %
        numeric_value = re.findall(r"[-+]?\d*\.\d+|\d+", answer_part)

        # If a numeric value is found, return it as a float
        if numeric_value:
            return float(numeric_value[0])
        else:
            return None  # Return None if no valid number is found
    except (IndexError, ValueError):
        return None  # Return None if there's an error

def solve_cot(problem: str) -> Optional[int]:
    resp = client.completions.create(
        model=model,
        temperature=0,
        prompt=prompt_cot(problem),
        max_tokens=150,
        stop=["Done"]
    )
    return extract_cot(resp.choices[0].text)

def accuracy_cot(problems: List[dict]) -> List[float]:
    accuracies = []

    for _ in tqdm(range(5)):
        total_correct = 0

        for item in problems:
            answer = solve_cot(item["question"])
            if answer == item["answer"]:
                total_correct += 1
        accuracies.append(total_correct / len(problems))

    return accuracies


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
