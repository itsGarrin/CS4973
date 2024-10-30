#imports
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re
import html2text


visited_urls = set()

def normalize_url(base_url, subpage_url):
    # remove hash and query string and index.html
    subpage_url = subpage_url.split("#")[0].split("?")[0].split("index.html")[0]
    if subpage_url.startswith("/"):
        return base_url + subpage_url
    return base_url + '/' + subpage_url

def get_text_with_headers(soup):
    text = ""

    for element in soup.children:
        print(element.name)
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text += f"\n{element.name.upper()}: {element.get_text()}\n"
        else:
            text += element.get_text()

    return text.strip()

# Function to retrieve and parse subpage URLs
def get_subpages(url):
    if url in visited_urls:
        return [] 
    visited_urls.add(url)
    subpages = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the relevant part of the HTML (e.g., the main content)
    # Adjust the selector based on the structure of the webpage you're scraping
    main_content = soup.find('main')  # or soup.find('div', class_='content'), etc.

    # If no specific element is found, use the entire body
    if not main_content:
        main_content = soup.find('body')

    # Convert the HTML to Markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    main_content_md = h.handle(str(main_content))

    subpages.append({"url": url, "content": main_content_md})

    # Extract links to subpages within the main page
    for link in soup.find_all("a", href=True):
        subpage_url = link["href"]
        if subpage_url.startswith(url.split(".edu")[1]): 
            if subpage_url.endswith(".pdf"):
                continue
            print(subpage_url) # Ensure it's a subpage
            sub_subpages = get_subpages(normalize_url(url.split(".edu")[0] + ".edu", subpage_url))
            subpages.extend(sub_subpages)
    return subpages

# Function to clean up content
def clean_content(content):
    return re.sub(r'\n\s*\n', '\n', content)  # Remove blank lines

# get_subpages("https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/bscs")
# response = requests.get("https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/bscs/#programrequirementstext")
# soup = BeautifulSoup(response.text, "html.parser")
# main_content_md = md(response.text)
# print(soup.get_text())
