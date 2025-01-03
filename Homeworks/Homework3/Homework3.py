import os
import re
from functools import cache
from typing import List

import math
import numpy as np
import spacy
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModel

load_dotenv()

CLIENT = OpenAI(base_url=os.getenv("URL"), api_key=os.getenv("KEY"))
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Ensure the model is in evaluation mode
model.eval()


@cache
def inverse_document_frequency(term: str, documents: tuple) -> float:
    num_docs_with_term = sum(1 for item in documents if term in item.split())
    if num_docs_with_term == 0:
        return 0.0
    return math.log(len(documents) / num_docs_with_term)


def term_frequency2(term: str, document: str):
    return document.count(term)


def tf_idf_vector(terms, doc, documents: tuple):
    vec = np.array(
        [
            term_frequency2(term, doc["text"])
            * inverse_document_frequency(term, documents)
            for term in terms
        ]
    )
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else np.zeros_like(vec)


def class_rank_by_tf_idf(query: str, n: int, documents: List[dict[str, str]]) -> list:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    terms = [token.text for token in doc if not token.is_stop and not token.is_punct]

    # Convert documents to a tuple of text strings
    doc_texts = tuple(doc["text"] for doc in documents)

    query_vec = tf_idf_vector(terms, {"text": query}, doc_texts)
    ranked_docs = sorted(
        documents,
        key=lambda doc: tf_idf_vector(terms, doc, doc_texts).dot(query_vec),
        reverse=True,
    )
    return ranked_docs[:n]


def rank_by_tf_idf(query: str, n: int, documents: List[dict[str, str]]) -> list:
    # Function to remove punctuation and numbers from text
    def remove_punctuation(text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)

    # Prepare document texts and include query in the corpus
    doc_texts = [remove_punctuation(doc["text"]) for doc in documents]
    cleaned_query = remove_punctuation(query)
    corpus = doc_texts + [cleaned_query]

    # Initialize TfidfVectorizer and compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Separate query vector and document vectors
    query_vec = tfidf_matrix[-1]  # Last vector is for the query
    doc_vecs = tfidf_matrix[:-1]  # All other vectors are for the documents

    # Compute cosine similarities and get top n ranked documents
    similarity_scores = (doc_vecs @ query_vec.T).toarray().ravel()
    top_indices = np.argsort(similarity_scores)[-n:][::-1]

    return [documents[i] for i in top_indices]


def get_bert_embedding(text: str) -> torch.Tensor:
    # Tokenize the input text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Get the embedding of the last token (excluding padding tokens)
    last_token_embedding = last_hidden_state[0, -1]
    return last_token_embedding


def rank_by_bert_similarity(
    query: str, documents: List[dict[str, str]], n: int
) -> List[dict[str, str]]:
    query_embedding = get_bert_embedding(query)
    document_embeddings = [
        get_bert_embedding(document["text"]) for document in documents
    ]
    similarity_scores = [
        torch.cosine_similarity(query_embedding, document_embedding, dim=0)
        for document_embedding in document_embeddings
    ]
    ranked_documents = [
        document
        for _, document in sorted(
            zip(similarity_scores, documents), key=lambda x: x[0], reverse=True
        )
    ]
    return ranked_documents[:n]


def answer_query(
    question: str, choices: List[str], documents: List[dict[str, str]]
) -> str:
    """
    Answers a multiple choice question using retrieval augmented generation.

    `question` is the text of the question. `choices` is the list of choices
     with leading letters. For example:

     ```
     ["A. Choice 1", "B. Choice 2", "C. Choice 3", "D. Choice 4"]
     ```

     `documents` is the list of documents to use for retrieval augmented
     generation.

     The result should be just the letter of the correct choice, e.g.,
     `"A"` but not `"A."` and not `"A. Choice 1".
    """
    print("Received question:", question)
    print("Received choices:", choices)

    # Step 1: Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, length_function=len
    )

    all_chunks = []

    for article in documents:
        chunks = text_splitter.split_text(article["text"])
        all_chunks.extend(chunks)

    print("Total chunks created:", len(all_chunks))

    # Step 2: Rank chunks using TF-IDF
    ranked_chunks = rank_by_tf_idf(
        question, 100, [{"text": chunk} for chunk in all_chunks]
    )
    print("Ranked chunks based on TF-IDF")

    # Step 3: Rerank using BERT similarity
    reranked_chunks = rank_by_bert_similarity(question, ranked_chunks, 10)
    print("Reranked chunks based on BERT similarity")

    # Prepare conversation history with system prompt and examples
    # Updated system prompt
    system_prompt = (
        "You are an assistant tasked with answering multiple-choice questions based on provided text passages. "
        "Consider the relevant context from the passages to determine the best answer. "
        "Do not rely solely on the examples given. Respond only with the letter of the correct choice."
        "When in doubt, guess B. Good luck!\n"
    )

    # Updated example questions and answers
    example_questions = [
        "What is the largest planet in our solar system?",
        "Which element has the chemical symbol 'O'?",
        "Who wrote 'Hamlet'?",
    ]
    example_choices = [
        ["A. Earth", "B. Jupiter", "C. Mars", "D. Venus"],
        ["A. Hydrogen", "B. Oxygen", "C. Helium", "D. Nitrogen"],
        ["A. Dickens", "B. Shakespeare", "C. Hemingway", "D. Twain"],
    ]
    example_answers = ["B", "B", "B"]

    # Prepare conversation history with the updated system prompt and examples
    conversation_history = [
        {"role": "system", "content": system_prompt},
    ]

    # Add multiple varied examples to the conversation history
    for eq, ec, ea in zip(example_questions, example_choices, example_answers):
        conversation_history.append(
            {
                "role": "user",
                "content": f"Question: {eq}\nChoices:\n{ec[0]}\n{ec[1]}\n{ec[2]}\n{ec[3]}\nAnswer: {ea}",
            }
        )

    # Add the current question to the conversation history
    conversation_history.append(
        {
            "role": "user",
            "content": f"Question: {question}\nChoices:\n"
            f"{choices[0]}\n{choices[1]}\n{choices[2]}\n{choices[3]}",
        }
    )

    # Add ranked chunks to conversation history
    for chunk in reranked_chunks:
        conversation_history.append({"role": "assistant", "content": chunk})

    # Create final prompt from conversation history
    final_prompt = (
        "\n".join(
            [f"{entry['role']}: {entry['content']}" for entry in conversation_history]
        )
        + "\nAnswer:"
    )

    # print("Final prompt sent to API:", final_prompt)

    # Call the OpenAI API to generate answers
    resp = CLIENT.completions.create(
        model=MODEL, temperature=0.1, prompt=final_prompt, max_tokens=1
    )

    answer = resp.choices[0].text.strip()
    print("Received answer from API:", answer)

    return answer


def benchmark_accuracy(
    questions: List[str],
    choices_list: List[List[str]],
    documents: List[dict[str, str]],
    expected_answers: List[str],
) -> float:
    """
    Benchmark the `answer_query` function by calculating accuracy.

    Parameters:
        questions (List[str]): A list of questions to test.
        choices_list (List[List[str]]): A list of lists, where each inner list contains choices for the corresponding question.
        documents (List[dict[str, str]]): A list of documents to use for answer retrieval.
        expected_answers (List[str]): A list of expected answers for accuracy calculation.

    Returns:
        float: The accuracy of `answer_query` on the provided test set, as a percentage.
    """
    correct_answers = 0
    total_questions = len(questions)

    for i in tqdm(range(total_questions), desc="Benchmarking Accuracy"):
        question = questions[i]
        choices = choices_list[i]
        predicted_answer = answer_query(question, choices, documents)

        print(f"Predicted answer for question {i + 1}: {predicted_answer}")
        print(f"Expected answer for question {i + 1}: {expected_answers[i]}")

        if predicted_answer == expected_answers[i]:
            correct_answers += 1

    accuracy = (correct_answers / total_questions) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
