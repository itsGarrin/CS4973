import praw

reddit = praw.Reddit(
    client_id="BrA1seVpmeQqXQWOIxnTdA",
    client_secret="o3Oj5eEIG8YAQZ-0NPVqE9ZkyyET3A",
    password="39clues",
    user_agent="fantasyfootballdata",
    username="joindaclub",
)

for submission in reddit.front.hot(limit=256):
    print(submission.score)




