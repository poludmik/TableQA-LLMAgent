"""
Goes through every page of documentation on https://pandas.pydata.org/docs/reference/api/, extracts text
"""

from bs4.element import NavigableString, Tag
import requests
from bs4 import BeautifulSoup
from tableqallmagent.logger import *
import pandas as pd

# The main API reference page
urls = [
"https://pandas.pydata.org/docs/reference/io.html",
"https://pandas.pydata.org/docs/reference/general_functions.html",
"https://pandas.pydata.org/docs/reference/series.html",
"https://pandas.pydata.org/docs/reference/frame.html",
"https://pandas.pydata.org/docs/reference/arrays.html",
"https://pandas.pydata.org/docs/reference/indexing.html",
"https://pandas.pydata.org/docs/reference/offset_frequency.html",
"https://pandas.pydata.org/docs/reference/window.html",
"https://pandas.pydata.org/docs/reference/groupby.html",
"https://pandas.pydata.org/docs/reference/resampling.html",
"https://pandas.pydata.org/docs/reference/style.html",
"https://pandas.pydata.org/docs/reference/plotting.html",
"https://pandas.pydata.org/docs/reference/options.html",
"https://pandas.pydata.org/docs/reference/extensions.html",
"https://pandas.pydata.org/docs/reference/testing.html",
"https://pandas.pydata.org/docs/reference/missing_value.html"
]

extracted_data = []


for url in urls:
    # Fetch the main page
    response = requests.get(url)
    html = response.text

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup.prettify())

    # Find all <a> tags within the section that contains the links to the subpages
    # This example assumes that all relevant links are within a specific div or section.
    # You might need to adjust the selector based on the actual structure of the webpage.
    links = soup.find_all('a', href=True)

    # Filter links: Example here filters for links that seem to point to API methods
    subpage_links = [link['href'] for link in links if link['href'].endswith('.html')]

    # Iterate through each subpage link
    for subpage_link in subpage_links:
        # Construct the full URL (assuming relative links)
        full_url = f'https://pandas.pydata.org/docs/reference/{subpage_link}'

        # Fetch the subpage
        subpage_response = requests.get(full_url)
        subpage_html = subpage_response.text



        borsch = BeautifulSoup(subpage_html, 'html.parser')

        method_name_section = borsch.find('h1')
        method_name = method_name_section.get_text(strip=True) if method_name_section else "Method name not found"


        # examples_section = borsch.find(lambda tag: tag.name == "p" and "Examples" in tag.text)
        # # If the Examples section header was found, we try to find the next `div` which usually contains the examples
        # if examples_section:
        #     examples_content = examples_section.find_next_sibling("div")
        #     examples_text = examples_content.get_text(strip=True) if examples_content else "Examples content not found"
        # else:
        #     examples_text = "Examples section not found"


        examples_heading = borsch.find(lambda tag: tag.name == "p" and "Examples" in tag.text)

        if examples_heading:
            content = []
            # Find all next siblings until the next section heading
            for sibling in examples_heading.find_next_siblings():
                # Check if we've reached the next section heading
                if sibling.name in ["h2", "h3"]:
                    break  # Stop collecting content
                if isinstance(sibling, (NavigableString, Tag)):
                    content.append(str(sibling))

            examples_html = "\n".join(content)
            # Now examples_html contains all HTML from the Examples section
            # You can further parse this HTML or extract text as needed
            examples_soup = BeautifulSoup(examples_html, 'html.parser')
            # Do something with examples_soup, like extracting text or code snippets
            code_snippets = examples_soup.find_all(['pre'])
            examples_text = "\n".join([snippet.get_text() for snippet in code_snippets])
            # examples_text = examples_soup.get_text("")
        else:
            examples_text = "Examples section not found"



        # You can parse the subpage HTML with BeautifulSoup again here
        # For simplicity, this example just prints the URL
        if "api/pandas" in full_url:
            print()
            print(f"{BLUE}{method_name}{RESET}")
            print(examples_text)
            method_name = method_name.split("#")[0] # remove '#' from method name
            extracted_data.append((method_name, examples_text))
        # time.sleep(0.1)
        # Here you can add more logic to process each subpage as needed


# df = pd.DataFrame(extracted_data, columns=["method_name", "examples"])

df = pd.read_excel("dataset/pandas_corpus/pandas_api_examples.xlsx")
# remove all '#' occurences from method names
df["method_name"] = df["method_name"].str.replace("#", "")
df.to_excel("dataset/pandas_corpus/pandas_api_examples.xlsx", index=False)

