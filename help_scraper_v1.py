import json
import requests
import openai
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OpenAI API key not found. Please set it in the .env file.")
    exit(1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_FOLDER = "scraped_data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def list_json_files():
    """
    Lists all available scraped JSON files.
    """
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".json")]
    return files

def validate_url(domain):
    """
    Validate the given help website URL.
    """
    domain = domain.replace("https://", "").replace("http://", "")  # Remove scheme
    pattern = r"^help\.[\w.-]+\.[a-z]{2,}$"
    if not re.match(pattern, domain):
        raise ValueError("Invalid domain. Ensure it follows the format 'help.something.com'.")
    return domain 

def scrape_help_docs(domain, max_pages=5):
    """
    Scrapes help documentation and stores it in a structured format.
    """
    base_url = f"https://{domain}"
    visited = set()
    documentation = []

    def crawl(url, depth):
        if depth >= max_pages or url in visited:
            return

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Skipping {url} due to error: {e}")
            return

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary elements
        for tag in soup(["header", "footer", "nav", "script", "style"]):
            tag.decompose()

        content = " ".join(soup.stripped_strings)

        if content:
            documentation.append({"url": url, "text": content})
            visited.add(url)

        # Find internal links
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            full_url = urljoin(base_url, href)
            if full_url not in visited and domain in full_url:
                crawl(full_url, depth + 1)

    crawl(base_url, 0)

    # Save documentation to JSON
    filename = os.path.join(DATA_FOLDER, f"{domain}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(documentation, f, indent=4, ensure_ascii=False)
    
    print(f" Successfully scraped and saved {len(documentation)} pages from {domain}")
    return filename

def load_docs_from_json(filename):
    """
    Loads stored help documentation from a JSON file.
    """
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"Error: JSON file {filename} not found!")
        return None

    print(f"Loading documentation from {filename}...")  # Debug log

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def create_faiss_index(docs):
    """
    Creates a FAISS index from the extracted documentation.
    """
    if not docs:
        print("Error: No documents found to index.")
        return None, [], []

    texts = [doc["text"] for doc in docs]
    urls = [doc["url"] for doc in docs]

    print(f"Indexing {len(texts)} documents...")  # Debug log

    try:
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)

        if embeddings is None or len(embeddings) == 0:
            print("Error: No embeddings were generated. FAISS indexing failed.")
            return None, [], []

        if embeddings.ndim == 1:
            print("Error: Embeddings have incorrect dimensions. FAISS indexing failed.")
            return None, [], []

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"FAISS index created with dimension: {dimension}")
        return index, texts, urls

    except Exception as e:
        print(f"Error during FAISS indexing: {e}")
        return None, [], []

def query_docs(query, texts, urls, index, top_k=3):
    """
    Finds the most relevant documentation chunks using FAISS similarity search.
    """
    if index is None:
        print("Error: FAISS index is not initialized.")
        return []

    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)

        if len(indices) == 0 or len(indices[0]) == 0:
            print(f"Debug: No FAISS results for query: {query}")
            return []

        relevant_docs = []
        for i in range(len(indices[0])):  
            if indices[0][i] < len(texts) and distances[0][i] < 2.0:  # Increased threshold
                relevant_docs.append((texts[indices[0][i]], urls[indices[0][i]]))

        print(f"Debug: Retrieved {len(relevant_docs)} relevant documents for query: {query}")

        # Print retrieved documents for debugging
        for doc in relevant_docs:
            print(f"Retrieved Doc: {doc[1]} - {doc[0][:300]}...\n")

        return relevant_docs

    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def ask_llm(question, texts, urls, index):
    """
    Uses OpenAI's GPT model to generate an answer based on retrieved documents.
    If no relevant document is found, returns: "This information is not found in the documentation."
    """
    retrieved_docs = query_docs(question, texts, urls, index)

    if not retrieved_docs:
        print(f"Debug: No relevant documents for query: {question}")
        return "This information is not found in the documentation."

    retrieved_texts = "\n".join([f"Source: {url}\n{text[:500]}..." for text, url in retrieved_docs])

    print(f"Debug: Sending the following text to OpenAI:\n{retrieved_texts[:1000]}...\n")  # Debug log

    prompt = f"""
    You are an AI assistant that provides answers **strictly based on the provided documentation**.
    Do not generate answers beyond the information retrieved from the documentation.
    
    **Help Documentation (Extracted from {len(retrieved_docs)} Sources):**
    {retrieved_texts}

    **Rules for Answering:**
    1. Only use the information from the provided documentation.
    2. If the information is missing, respond with: "This information is not found in the documentation."
    3. Cite the sources (links) when giving answers.
    
    **User Question:**
    {question}

    **AI Answer:**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a documentation-based AI assistant. Only answer using the provided text."},
                {"role": "user", "content": prompt}
            ]
        )

        print("Debug: OpenAI API was successfully called.")  # Debug log
        return response.choices[0].message.content

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return "Error: Could not process the response from OpenAI."

    except Exception as e:
        print(f"Unexpected error while processing LLM response: {e}")
        return "Error: An unexpected issue occurred while generating a response."


if __name__ == "__main__":
    print("Help Documentation AI Assistant")
    
    while True:
        json_files = list_json_files()
        if json_files:
            print("\nAvailable JSON Files:")
            for idx, file in enumerate(json_files, 1):
                print(f"{idx}. {file}")

        domain = input("\nEnter the help site domain or select a JSON file number: ").strip()

        if domain.isdigit():
            selected_file = json_files[int(domain) - 1]
            docs = load_docs_from_json(selected_file)
        else:
            try:
                validated_domain = validate_url(domain)
                json_file = scrape_help_docs(validated_domain)
                docs = load_docs_from_json(json_file)
            except ValueError as e:
                print(f"Error: {e}")
                continue

        if not docs:
            print("Error: No documentation found. Please scrape it first.")
            continue

        faiss_index, texts, urls = create_faiss_index(docs)
        if not faiss_index:
            print("Error: FAISS index creation failed.")
            continue

        while True:
            user_question = input("\nAsk a question (or type 'exit' to return to main menu): ").strip()
            if user_question.lower() == "exit":
                break

            print("\nAnswer:", ask_llm(user_question, texts, urls, faiss_index))