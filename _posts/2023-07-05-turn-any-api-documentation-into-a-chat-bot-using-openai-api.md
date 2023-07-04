---
layout: post
title:  "Turn Any API Documentation into a Chat Bot Using OpenAI API"
author: nauroz
categories: [ openai, chat gpt, development ]
image: assets/images/pexels-christina-morillo-1181271.jpg
description: "Discover how to transform API documentation into a responsive chatbot using OpenAI API."
featured: true
---
# Turning Any API Documentation into a Chat Bot Using OpenAI API

Status: Backlog
Visuals: No

Navigating API documentation can often feel like venturing into a maze, especially for those unfamiliar with the intricacies of programming. Fortunately, OpenAI's API offers a solution by enabling us to transform any API documentation into an easy-to-use, interactive chatbot. This article illuminates how to implement this and outlines the significant advantages it provides.

## Unveiling the Benefits of a Chatbot for API Documentation

Transforming API documentation into a chatbot ushers in a myriad of benefits. Foremost, it offers users an engaging, intuitive, and interactive avenue for learning and utilizing the API. Instead of plowing through verbose documentation, users can now interact with the chatbot, simplifying the learning process and eliminating intimidation. This interactive approach enhances user engagement and promotes the wider adoption of the API. Furthermore, a chatbot delivers real-time support and troubleshooting, swiftly assisting users to overcome any encountered issues.

## Getting started

We'll begin by importing the necessary libraries.

```python
import openai
import tiktoken
import pinecone
import re
import PyPDF2
import pandas as pd
from tqdm.auto import tqdm
```

Next, let's initialize OpenAI and Pinecone libraries.

```python
openai.api_key = "<openai-api-key>"
pinecone.init(api_key="<pinecone-api-key>", environment="<pinecone-api-env>")
```

## Text Extraction

### From PDF Documentation

In cases where the API documentation is available in a different format like Markdown, using a parser like CommonMark to transform it into a more accessible format like HTML is beneficial. Consequently, a web scraping tool can be employed to extract the text, which is then leveraged to train the chatbot using OpenAI's API. This process, however, may demand additional text cleaning and formatting to optimize it for chatbot training.

First, let's create a function to cleanse the text from symbols like **`\n`** and remove any extra spaces.

```python
def remove_newlines(text):
    return re.sub(r'[\n\t]+', ' ', text).strip()
```

We'll then utilize **`PyPDF2`** to retrieve the text from a PDF file. Let's construct that function.

```python
def read_pdf_file(file_name):
    pdfFileObj = open(file_name, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    count = 0
    pages = []

    while count < len(pdfReader.pages):
        pageObj = pdfReader.pages[count]
        count += 1
        pages.append(clean_text(pageObj.extract_text()))
    return pages

api_text = read_pdf_file("api-doc.pdf")

content_df = pd.DataFrame(api_text, columns=['text'])
```

The **`read_pdf_file`** function reads a PDF file and extracts its textual content. It takes in one argument, **`file_name`**, which is the name of the PDF file to read.

After opening the PDF file in binary mode using the **`open()`** function, a **`PdfReader`** object is created using the **`PyPDF2`** library. A counter variable **`count`** and an empty list **`pages`** are initialized to track the number of pages processed and to store the text content of each page, respectively.

The function then initiates a while loop that continues until all the pages in the PDF file are processed. The **`extract_text()`** method of the **`pageObj`** object (a **`PageObject`** instance representing the current page) is used to extract the text content, which is then cleansed of newline characters using the **`remove_newlines()`** function before appending it to the **`pages`** list.

Finally, the **`pages`** list, which contains the textual content of each page in the PDF file, is returned by the function.

The function **`read_pdf_file()`** is then invoked with the name of a PDF file called "api-doc.pdf", and the resulting textual content is stored in the **`api_text`** variable.

### From Web-hosted Documentation

For API documents hosted on a website, **`BeautifulSoup`** can be employed to crawl the site and extract all necessary text. Ensure to import **`BeautifulSoup`** and **`urljoin`** libraries first, then apply the following code.

```python
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests

web_url = "https://www.twilio.com/docs"

visited_urls = set()
pages = []

def crawl_url(url):
    print("crawling", url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    title_tag = soup.title
    text = soup.get_text()
    title = title_tag.string if title_tag is not None else ''
    page = {"url": url, "title": title, "text": remove_newlines(text)}
    pages.append(page)

    for link in soup.find_all("a", href=True):
        href = link["href"]
        absolute_url = urljoin(web_url, href)
        crawled_pages = [page["url"] for page in pages if page["url"] == absolute_url]

        if absolute_url.startswith(web_url) and absolute_url not in visited_urls and \
        len(crawled_pages) == 0:
            visited_urls.add(absolute_url)
            crawl_url(absolute_url)

crawl_url(web_url)

content_df = pd.DataFrame(pages, columns = ["url", "title","text"])
```

The **`crawl_url`** function crawls a specified URL and extracts its content. It accepts one argument, **`url`**, representing the URL to crawl.

Upon sending a GET request to the URL using the **`requests`** library, the HTML content of the page is parsed using **`BeautifulSoup`**. The title and text content of the page are then extracted and cleansed of newlines.

The function constructs a dictionary, **`page`**, to store the URL, title, and text content of the page, and adds this dictionary to the **`pages`** list, which records all the previously crawled pages.

Subsequently, the function locates all the links on the page and for each link, extracts the **`href`** attribute, transforming it into an absolute URL.

The function then checks if the absolute URL is a part of the base URL, has not been previously visited, and has not been crawled before. If all these conditions are met, the function adds the absolute URL to the set of visited URLs and recursively calls itself with the absolute URL as an argument.

## Unveiling Tokenization

With tokenization we can fragment the extracted API text into smaller, more manageable chunks or 'tokens.' This methodological procedure ensures that the resulting tokens fit perfectly within the maximum limit permitted by the OpenAI API when it is processing prompts.

```python
tokenizer = tiktoken.get_encoding("cl100k_base")

content_df["tokenized"] = content_df["text"].apply(lambda x: len(tokenizer.encode(x)))
```

To visualize our token distribution, we can plot a histogram. Depending on the nature and volume of your API documentation, the graph might appear differently.

```python
content_df["tokenized"].hist()
```

![output.png](Turning%20Any%20API%20Documentation%20into%20a%20Chat%20Bot%20Usin%2031ff36b40f0c48129f1638e8d0d7abb6/output.png)

Let's suppose our token limit is capped at 500. To accommodate this, we'll create a function that slices text into smaller chunks whenever it surpasses this limit.

```python
max_tokens = 500

def split_text(text):
    if len(tokenizer.encode(text)) > max_tokens:
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    else:
        return [text]
```

The **`split_text`** function divides a given text into several sections, each with a maximum length of 500 tokens. Should the length of the encoded text exceed 500 tokens, the function carves the text into smaller slices of 500 tokens each. However, if the encoded text's length falls within or precisely at the limit, the function returns the text as is.

By iterating through each row of our data and applying this function, we can create a new dataframe with reduced text segments. Finally, we calculate the tokenized length of each segment.

```python
# Create a new dataframe with shortened text. Make sure to match your column names.
shortened = [{"title": row["title"], "text": chunk} for _, row in content_df.iterrows() for chunk in split_text(row["text"]) if row["text"] is not None]

# Create a new dataframe with tokenized length
tokenized_df = pd.DataFrame(shortened, columns=["title", "text"])
tokenized_df["tokenized"] = tokenized_df["text"].apply(lambda x: len(tokenizer.encode(x)))
```

Now, by plotting a histogram of our tokens' length, we can confirm all tokens align within the 500-token range.

```python
tokenized_df["tokenized"].hist()
```

![output2.png](Turning%20Any%20API%20Documentation%20into%20a%20Chat%20Bot%20Usin%2031ff36b40f0c48129f1638e8d0d7abb6/output2.png)

## Embeddings

Embeddings serve as vectors in a high-dimensional space, symbolizing words or phrases. In the scope of language models like OpenAI's GPT, embeddings are instrumental in translating words or phrases into numerical representations that encapsulate their semantic meaning. These representations prove valuable in various tasks such as language translation or sentiment analysis.

Vector databases such as Pinecone are potent tools for storing and querying these embeddings efficiently. This capability facilitates swift and precise retrieval of similar words or phrases based on their semantic meaning.

To leverage this power, let's generate embeddings of our text and store them in Pinecone. To circumvent possible function failure due to OpenAI's rate limits for the Embeddings API, we employ a try/catch block coupled with a timeout.

```python
import time

def create_embedding(x):
    try:
        print(".")
        return openai.Embedding.create(input=x, engine="text-embedding-ada-002")["data"][0]["embedding"]
    except openai.error.RateLimitError:
        print("API limit reached. Waiting for 30 seconds before trying again...")
        time.sleep(40)
        return create_embedding(x)

tokenized_df["embeddings"] = tokenized_df.text.apply(create_embedding)
```

Finally, let's assign unique identifiers (IDs) to each row of our dataframe. This allows us to maintain robust tracking and organization within our data.

```python
from uuid import uuid4

tokenized_df["id"] = [str(uuid4()) for _ in range(len(tokenized_df))]
```

## Importing into Pinecone

Let's dive into the process of integrating our data into Pinecone, a vector database.

```python
# Set up Pinecone index
index_name = 'my-new-api-index'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric='dotproduct')
index = pinecone.Index(index_name)

# Define batch size for embeddings
batch_size = 100

# Convert DataFrame to list of dictionaries
chunks = tokenized_df.to_dict(orient='records')

# Upsert embeddings into Pinecone in batches of 100
for i in tqdm(range(0, len(chunks), batch_size)):
    i_end = min(len(chunks), i + batch_size)
    meta_batch = chunks[i:i_end]
    ids_batch = [x['id'] for x in meta_batch]
    embeds = [x['embeddings'] for x in meta_batch]
    meta_batch = [{'text': x['text']} for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    index.upsert(vectors=to_upsert)

index.describe_index_stats()
```

This code sets up a Pinecone index, converts a Pandas DataFrame to a list of dictionaries, and upserts embeddings into the Pinecone index in batches.

The code first defines a variable called `index_name` with the name of the Pinecone index to create or use. It checks if the index already exists using the `pinecone.list_indexes()` method, and creates the index with the specified name and parameters if it doesn't exist using the `pinecone.create_index()` method.

The code then defines a variable called `batch_size` with the number of embeddings to upsert in each batch. It converts the Pandas DataFrame called `tokenized_df` to a list of dictionaries using the `to_dict()` method with the `orient="records"` parameter.

The code then enters a loop that iterates over the chunks of embeddings in batches of `batch_size`. For each batch, it extracts the IDs, embeddings, and metadata from the list of dictionaries, and creates a list of tuples containing the ID, embedding, and metadata for each chunk. It then calls the `index.upsert()` method to upsert the embeddings into the Pinecone index.

Finally, the code calls the `index.describe_index_stats()` method to print out some statistics about the Pinecone index, such as the number of vectors and the dimensionality of the embeddings.

This will yield output information regarding your new index, as seen below:

```json
{
	"dimension": 1536,
	"index_fullness": 0.0,
	"namespaces": { "": { "vector_count": 579 } },
	"total_vector_count": 579
}
```

## Extracting Relevant Excerpts from Pinecone

The OpenAI retriever plugin becomes invaluable at this point as it extracts relevant information from the Pinecone database we've just created. It allows us to craft a prompt in natural language, and it will trawl through the API documentation embedded in Pinecone to provide an answer.

```json
model = "text-embedding-ada-002"
input = "How do I send a text?"

query = openai.Embedding.create(
    input=user_input,
    engine=model
)

query_embeddings = query['data'][0]['embedding']

response = index.query(query_embeddings, top_k=5, include_metadata=True)
```

With the context for our question at ready, we create an "augmented" query by concatenating all the context along with the question. We then forward this to ChatGPT to create a clear answer.

```json
# Extract the "metadata" field from each match in the response and store it in a list called "contexts"
contexts = [item["metadata"]["text"] for item in response["matches"]]

# Join the "contexts" list together into a single string with "---" between each context and "-----" at the end
augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + input

# Print the final string
print(augmented_query)
```

## Getting ChatGPT to Create the Answer in Plain English

We now introduce the ChatGPT plugin to our code and pass the **`augmented_query`** to it, yielding an easy-to-understand answer in clear English. By using the system message option in the ChatGPT plugin, we assign it a "developer copilot" role. Its mission? To help developers understand the Twilio API.

```json
chat = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a developer copilot helping the team to understand Twilio API and answer questions."},
        {"role": "user", "content": augmented_query}
    ]
)

print(chat["choices"][0]["message"]["content"])
```

You should now see an answer in plain English, which is both easily understandable and concise. It might look something like this:

---

To send a text message using the Twilio API, you'll need to use the `messages.create()` method in the Twilio client. Below is an example:

```python
from twilio.rest import Client

# Your Twilio account SID and Auth Token
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'

client = Client(account_sid, auth_token)

message = client.messages.create(
    body='Hello, this is the message body.',
    from_='Twilio_Number', # This is your Twilio number
    to='Recipient_Number' # This is the phone number the SMS goes to
)

print(message.sid)
```

This code is written in Python. The `body` attribute is the actual message you want to send. `from_` is your Twilio number and `to` is the recipient's phone number. A successful execution will print out the message SID which can be used to reference the message in the future.

Remember to replace `'your_account_sid'`,`'your_auth_token'`, `'Twilio_Number'`,`'Recipient_Number'` with your actual account SID, auth token, your Twilio number and recipient's number.

Make sure you've installed the Twilio Python library. You can install it using pip:

```python
pip install twilio
```

For other languages such as Node.js, PHP, Ruby or Java, you can refer to Twilio's API documentation for the equivalent code. Twilio SDKs available for these and also C#, .NET and Swift.

---

## Conclusion

To sum it up, converting API documentation into a chatbot using the OpenAI API can bring significant benefits, such as enhanced user engagement and real-time support. The steps outlined above allow you to develop a chatbot that is both user-friendly and effective. So why not take the plunge and explore how this could enrich your API documentation?