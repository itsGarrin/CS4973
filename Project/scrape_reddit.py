import re

import praw
from tqdm import tqdm

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

def scrape_thread_content(thread_url):
    # Extract thread ID from the URL
    thread_id = thread_url.split("/")[-3]
    thread = reddit.submission(id=thread_id)

    # Extract thread details
    thread_content = {
        "title": thread.title,
        "url": thread.url,
        "author": thread.author.name if thread.author else "Deleted",
        "comments": [],
    }

    # Scrape comments
    thread.comments.replace_more(limit=None)
    for comment in thread.comments.list():
        if hasattr(comment, "body"):
            thread_content["comments"].append(comment.body)

    return thread_content


def scrape_daily_post_threads(post_id):
    # Get all links from the daily post
    links = extract_links_from_post(post_id)
    print(f"Found {len(links)} threads.")

    threads_data = {}
    for link in links:
        thread_data = scrape_thread_content(link)
        threads_data[thread_data["title"]] = thread_data

    return threads_data


# Usage Example
if __name__ == "__main__":
    daily_post_id = "1h1uirv"  # Replace with the daily post's ID
    threads = scrape_daily_post_threads(daily_post_id)

    # Output results
    for title, data in tqdm(threads.items()):
        print(f"\nThread Title: {title}")
        print(f"URL: {data['url']}")
        print(f"Author: {data['author']}")
        print(f"Number of Comments: {len(data['comments'])}")
        print("Sample Comments:")
        for comment in data["comments"][:3]:  # Show first 3 comments
            print(f"- {comment}\n")
