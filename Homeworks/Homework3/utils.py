import spacy 
from typing import List
import datasets
import math
from functools import cache
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from openai import OpenAI
import os 

load_dotenv()

CLIENT = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

wikipedia_dataset = load_dataset("nuprl/engineering-llm-systems", name="wikipedia-northeastern-university", split="test")
obscure_questions_dataset = load_dataset("nuprl/engineering-llm-systems", name="obscure_questions", split="test")
obscure_questions_dataset_tiny = load_dataset("nuprl/engineering-llm-systems", name="obscure_questions", split="tiny")

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Ensure the model is in evaluation mode
model.eval()

@cache
def inverse_document_frequency(term: str) -> float:
    num_docs_with_term = 0
    for item in wikipedia_dataset:
        if term in item["text"].split():
            num_docs_with_term += 1
    if num_docs_with_term == 0:
        return 0
    return math.log(len(wikipedia_dataset) / num_docs_with_term)

@cache
def term_frequency2(term:str, document: str):
    return document.count(term)

# def tf_idf_vector0(terms, doc):
#     return np.array([term_frequency2(term, doc['text']) * inverse_document_frequency(term) for term in terms])

def tf_idf_vector(terms, doc):
    vec = np.array([term_frequency2(term, doc['text']) * inverse_document_frequency(term) for term in terms])
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)  # Return zero vector if all terms are absent
    return vec / norm

def rank_by_tf_idf(query: str, n: int) -> list:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    terms = [token.text for token in doc if not token.is_stop and not token.is_punct]
    print(terms)
    query_vec = tf_idf_vector(terms, { "text": query })
    ranked_docs = sorted(
        wikipedia_dataset,
        key=lambda doc: tf_idf_vector(terms, doc).dot(query_vec)  ,
        reverse=True
    )
    return ranked_docs[:n]

def get_bert_embedding(text: str) -> torch.Tensor:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state
    
    # Get the embedding of the last token (excluding padding tokens)
    last_token_embedding = last_hidden_state[0, -1]
    return last_token_embedding

def rank_by_bert_similarity(query: str, documents: List[str], n: int) -> List[str]:
    query_embedding = get_bert_embedding(query)
    document_embeddings = [get_bert_embedding(document["text"]) for document in documents]
    similarity_scores = [torch.cosine_similarity(query_embedding, document_embedding, dim=0) for document_embedding in document_embeddings]
    ranked_documents = [document for _, document in sorted(zip(similarity_scores, documents), reverse=True)]
    return ranked_documents[:n]

def answer_query(question: str, choices: List[str], documents: List[str]) -> str:
    """
    Answers a multiple choice question using retrieval augmented generation.

    `question` is the text of the question. `choices` is the list of choices
     with leading letters. For example:

     ```
     ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
     ```

     `documents` is the list of documents to use for retrieval augmented
     generation.

     The result should be the just the letter of the correct choice, e.g.,
     `"A"` but not `"A."` and not `"A. Choice 1"`.
     """
    print(question)
    # get the top 100 documents by tf-idf
    documents = rank_by_tf_idf(question, 100)
    # print([document["title"] for document in documents])

    # rerank the documents using BERT embeddings
    documents = rank_by_bert_similarity(question, documents, 20)

    # TODO: CHUNK PROPERLY

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=200,
    length_function=len)

    all_chunks = []
    for article in documents:
        chunks = text_splitter.split_text(article["text"])
        all_chunks.extend(chunks)
    
    print(len(all_chunks), all_chunks, all_chunks[0], len(all_chunks[0]))

    # TODO: FIGURE OUT HOW TO CALL IT HERE?
    # add the context to the question
    PROMPT = f"Context: {all_chunks[0]}\nQuestion: {question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    # PROMPT = f"Question: {question}\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    resp = CLIENT.completions.create(
        model=MODEL,
        temperature=0.1,
        prompt=PROMPT,
        max_tokens=1
    )
    
    # call the OpenAI API to generate answers
    return resp.choices[0].text.strip()

# Benchmark 'answer_query' using a subset of obscure questions.
# This might include iterating over the dataset and comparing outputs to the correct answers.
print(answer_query(obscure_questions_dataset_tiny[0]["prompt"], obscure_questions_dataset_tiny[0]["choices"], wikipedia_dataset))

# TERMS = ['NFL', 'team', 'drafted', 'William', 'Jeffrey', 'Thomas', 'round', '1972', 'NFL', 'Draft']
# print([inverse_document_frequency(term) for term in TERMS])

# query_vec = tf_idf_vector(TERMS, { "text": obscure_questions_dataset_tiny[1]["prompt"] })
# tf_idf_vector_actual = tf_idf_vector(TERMS, wikepedia_dataset[1091])
# tf_idf_vector_bad = tf_idf_vector(TERMS, wikepedia_dataset[389])
# print(tf_idf_vector_actual, tf_idf_vector_bad)

# print("\n")
# print(tf_idf_vector_actual.dot(query_vec))
# print(tf_idf_vector_bad.dot(query_vec))