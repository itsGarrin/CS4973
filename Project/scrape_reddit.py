import re

import pandas as pd
import praw

reddit = praw.Reddit(
    client_id="BrA1seVpmeQqXQWOIxnTdA",
    client_secret="o3Oj5eEIG8YAQZ-0NPVqE9ZkyyET3A",
    password="39clues",
    user_agent="fantasyfootballdata",
    username="joindaclub",
)


def extract_links_from_post(submission_id):
    submission = reddit.submission(id=submission_id)

    # Full selftext
    full_text = submission.selftext

    # Regex pattern to match markdown links starting with "Official:"
    pattern = r'\[Official:.*?\]\((/r/fantasyfootball/comments/[^\)]+)\)'
    matches = re.findall(pattern, full_text)

    # Prepend "https://reddit.com" to convert relative links to absolute URLs
    links = [f"https://reddit.com{match}" for match in matches]

    return links


def collect_answers_from_comment(comment, question_id):
    answers = []

    # If the comment is a reply to the question, or any other comment, capture it as an answer
    if comment.parent_id == f"t1_{question_id}" and comment.score >= 0:
        # Skip deleted comments
        if comment.body.lower() == "[deleted]" or not comment.author:
            return []

        answers.append({
            "answer": comment.body,
            "author": comment.author.name if comment.author else "Deleted"
        })

    # If there are replies to this comment, recursively collect them
    for reply in comment.replies:
        answers.extend(collect_answers_from_comment(reply, question_id))

    return answers


def scrape_thread_content(thread_url):
    # Extract thread ID from the URL
    thread_id = thread_url.split("/")[-3]
    thread = reddit.submission(id=thread_id)

    # Extract thread details
    thread_content = {
        "title": thread.title,
        "url": thread.url,
        "author": thread.author.name if thread.author else "Deleted",
        "qa_pairs": [],
    }

    # Scrape comments
    thread.comments.replace_more(limit=None)
    comments = list(thread.comments.list())

    # Debugging: Check how many comments we fetched
    print(f"Fetched {len(comments)} comments for thread: {thread.title}")

    # Iterate over all comments and treat only parent comments as questions
    for comment in comments:
        # Skip deleted comments
        if comment.body.lower() == "[deleted]" or not comment.author:
            continue

        if comment.score >= 0 and comment.parent_id == f"t3_{thread_id}":  # Only include parent comments (questions)
            question = comment.body
            question_author = comment.author.name if comment.author else "Deleted"

            # Collect answers to the question
            answers = collect_answers_from_comment(comment, comment.id)

            # If there are any answers, add the question and answers to the list
            if answers:
                thread_content["qa_pairs"].append({
                    "question": question,
                    "question_author": question_author,
                    "answers": answers
                })

    # Debugging: Check collected Q&A pairs
    if thread_content["qa_pairs"]:
        print(f"Collected {len(thread_content['qa_pairs'])} Q&A pairs.")
    else:
        print("No Q&A pairs found.")

    return thread_content


def scrape_daily_post_threads(post_id):
    # Get all links from the daily post
    links = extract_links_from_post(post_id)
    print(f"Found {len(links)} threads.")

    threads_data = []
    for link in links:
        thread_data = scrape_thread_content(link)

        # Store Q&A pairs in a structured format for DataFrame
        for qa_pair in thread_data["qa_pairs"]:
            question = qa_pair["question"]
            question_author = qa_pair["question_author"]
            for answer in qa_pair["answers"]:
                threads_data.append({
                    "thread_title": thread_data["title"],
                    "thread_url": thread_data["url"],
                    "question": question,
                    "question_author": question_author,
                    "answer": answer["answer"],
                    "answer_author": answer["author"]
                })

    # Debugging: Check the structure of threads_data before creating DataFrame
    if threads_data:
        print(f"Collected {len(threads_data)} Q&A pairs.")
    else:
        print("No data found.")

    return threads_data


# Usage Example
if __name__ == "__main__":
    daily_post_id = "1h1uirv"  # Replace with the daily post's ID
    threads = scrape_daily_post_threads(daily_post_id)

    # Convert the results into a pandas DataFrame
    df = pd.DataFrame(threads)

    # Debugging: Check the DataFrame contents
    if not df.empty:
        print(df.head())
    else:
        print("No data found.")

    # Save to parquet
    df.to_parquet("fantasy_football_qa_pairs.parquet", index=False)
