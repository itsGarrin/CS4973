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
        return f"""
    Input: Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?

    Reasoning:
    def reason_problem():
        num_pizzas = 7
        slices_per_pizza = 8
        total_slices = num_pizzas * slices_per_pizza
        num_people = 1 + 3
        slices_per_person = total_slices / num_people
        return slices_per_person

    Done

    Input: Sarah buys three times as many green pens as blue pens. The green pens cost 25% more than the blue pens. She spent $180 on blue pens that cost $30 each. How much did she spend on pens?

    Reasoning:
    def reason_problem():
        blue_pen_price = 30
        blue_pen_cost = 180
        green_pen_multiplier = 3
        green_pen_price_increase = 1.25
        num_blue_pens = blue_pen_cost // blue_pen_price
        num_green_pens = num_blue_pens * green_pen_multiplier
        green_pen_price = blue_pen_price * green_pen_price_increase
        total_blue_pen_cost = num_blue_pens * blue_pen_price
        total_green_pen_cost = num_green_pens * green_pen_price
        total_spent = total_blue_pen_cost + total_green_pen_cost

        return total_spent

    Done

    Input: Nikhil ordered one turkey meal that costs $14, 5 packs of milk that costs $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Nikhil paid a total of $50. How many boxes of pizza did Nikhil order if each box costs $8.50?

    Reasoning:
    def reason_problem():
        num_turkey_meal = 1
        turkey_meal_cost = 14
        milk_cost = 3
        num_packs_milk = 5
        apple_cost = 1.5
        num_apples = 4
        total_cost = 50
        money_left_over = (turkey_meal_cost * num_turkey_meal) + (milk_cost * num_packs_milk) + (apple_cost * num_apples)
        pizza_cost = 8.5
        boxes_pizza_ordered = money_left_over / pizza_cost
        return boxes_pizza_ordered

    Done

    Input: A school has 200 students. They are divided into 4 classes equally. If 10 students from each class join a sports event, how many students are left in the classes?

    Reasoning:
    def reason_problem():
        num_students = 200
        num_classes = 4
        num_students_per_class = num_students / num_classes
        num_students_left_per_class = num_students_per_class - 10
        total_students_left = num_students_left_per_class * 4
        return total_students_left

    Done

    Input: {problem}
    Reasoning:"""

def extract_pal(completion: str) -> Optional[int]:
    try:
        # Look for the answer by splitting at "Answer:" and extracting the part after that
        answer_part = completion.split("Reasoning:")[-1].strip()

        exec(answer_part)
        numeric_value = eval("reason_problem()")

        # If a numeric value is found, return it as a float
        if numeric_value:
            return float(numeric_value)
        else:
            return None  # Return None if no valid number is found
    except:
        print("BIG ISSUE\n")
        return None  # Return None if there's an error

def solve_pal(problem: str) -> Optional[int]:
    resp = client.completions.create(
    model=model,
    temperature=0,
    prompt=prompt_pal(problem),
    max_tokens=150,
    stop=["Done"])

    return extract_pal(resp.choices[0].text)


def accuracy_pal(problems: List[dict]) -> float:
    accuracies = []

    for _ in tqdm(range(5)):
        total_correct = 0

        for item in problems:
            answer = solve_pal(item["question"])
            if answer == item["answer"]:
                total_correct += 1
        accuracies.append(total_correct / len(problems))

    return accuracies
