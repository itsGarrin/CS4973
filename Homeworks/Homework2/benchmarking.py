import os

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from Homework2 import eval_agent, load_flights_dataset

load_dotenv()

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))


def load_yaml(p):
    with open(p):
        return yaml.safe_load(p)

all_benchmarks = [ load_yaml(f) for f in os.listdir() if f.endswith(".yaml") ]
for benchmark in all_benchmarks:
    print(benchmark)
    score = eval_agent(client, benchmark, load_flights_dataset()).score
    print("\n")
    print(score)
