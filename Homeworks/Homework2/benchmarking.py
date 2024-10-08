import yaml
from Homework2 import eval_agent, load_flights_dataset
from openai import OpenAI
import os

client = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))


def load_yaml(p):
    with open(p):
        return yaml.safe_load(p)

# all_benchmarks = [ load_yaml(f) for f in os.listdir() if f.endswith(".yaml") ]
all_benchmarks = ["benchmark7.yaml", "benchmark5.yaml", "benchmark4.yaml", "benchmark1.yaml", "benchmark2.yaml", "benchmark6.yaml", "example1.yaml", "benchmark3.yaml"]

for benchmark in all_benchmarks:
    print(benchmark)
    print("\n\n")
    print(eval_agent(client, benchmark, load_flights_dataset()).score)
    print("\n\n")