#imports
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re
import pandas as pd

# Function to retrieve and parse subpage URLs
def get_subpages(url):
    subpages = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Convert main page content to markdown
    main_content_md = md(response.text)
    subpages.append({"url": url, "content": main_content_md})

    # Extract links to subpages within the main page
    for link in soup.find_all("a", href=True):
        subpage_url = link["href"]
        if subpage_url.startswith(url):  # Ensure it's a subpage
            subpage_response = requests.get(subpage_url)
            subpage_md = md(subpage_response.text)
            subpages.append({"url": subpage_url, "content": subpage_md})
    return subpages

# Function to clean up content
def clean_content(content):
    return re.sub(r'\n\s*\n', '\n', content)  # Remove blank lines

