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