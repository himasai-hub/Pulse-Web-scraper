AI Help Documentation Scraper & Question Answering Model

Author: Hima Sai Eswaraka
Email: himasai197@gmail.com

This is an AI-powered help documentation assistant that allows users to:

Scrape and store documentation from help.something.com.
Ask questions about scraped websites, retrieving relevant answers.
Utilize FAISS for similarity search and GPT-4 for answering.
How It Works

User is given an option to:
Scrape a new website.
Query from already scraped websites.
If querying an existing website:
A list of available JSON files is displayed.
User selects a file and starts asking questions.
If the query is relevant to the document, the AI provides an answer with a source link.
If no relevant information is found, it returns:
 "This information is not found in the documentation."
If scraping a new website:
User must enter a URL in the correct format:
 help.something.com
If incorrect, the system rejects the URL and provides an example.

Limitations

Only one website can be scraped at a time.
Some non-open-source sites block scraping, leading to 0 pages found.
Limited time (24 hours) to complete this task means:
More advanced embeddings (e.g., base-dot-v1) could not be implemented.
No time for hyperparameter tuning or experimentation with different models.
 API & Technologies Used

Technology	Purpose
OpenAI GPT-4 / GPT-4o-mini	Answers user queries based on retrieved documents.
FAISS (Facebook AI Similarity Search)	Stores embeddings and retrieves relevant documents efficiently.
SentenceTransformer (all-MiniLM-L6-v2)	Converts text into numerical vectors (embeddings).
 Why This Combination?

Component	Function
all-MiniLM-L6-v2	Converts text into embeddings (vectors) for efficient search.
FAISS	Stores and retrieves similar embeddings quickly.
GPT-4	Generates answers based on the retrieved documentation.

Setup Instructions

Step 1: Install Dependencies
pip install openai python-dotenv requests beautifulsoup4 faiss-cpu sentence-transformers
 
 Step 2: Add Your OpenAI API Key
Create a .env file in the project directory:

OPENAI_API_KEY=your_openai_api_key_here
Step 3: Run the Program
python help_scraper_v1.py
 
 Example Output

Available JSON Files:
1. help.zluri.com.json
2. help.zoho.com.json
3. help.drop.com.json
4. help.dropbox.com.json

Enter the help site domain or select a JSON file number: 4

Loading documentation from help.dropbox.com.json...
Indexing 144 documents...
FAISS index created with dimension: 384

Ask a question (or type 'exit' to return to main menu):
 Example: Scraping a New Website
Enter the help site domain or select a JSON file number: help.uber.com 
Skipping https://help.uber.com due to error: 404 Client Error: Not Found for URL: https://help.uber.com/
 Successfully scraped and saved 10 pages from help.uber.com
 
 Example: Querying Help Documentation

Ask a question (or type 'exit' to return to main menu'): What is Dropbox?

Answer: The provided documentation does not contain specific information describing Dropbox. However, it does mention Dropbox Business, a plan for small, agile teams that allows you to edit and send out a project proposal, sign a contract, and deliver final files without switching between apps [Source](https://help.dropbox.com/plans/dropbox-business). It also refers to the Dropbox desktop app and system requirements needed for it. The required version for a Windows computer is Windows 10 or 11, not in S mode [Source](https://help.dropbox.com/installs/system-requirements).
 
 Notes

Ensure your .env file contains a valid OpenAI API key before running the script.
You can modify the script to use alternative AI models (e.g., Cohere, Hugging Face).
FAISS can be configured to run on GPUs for larger-scale searches.
 
 Future Improvements

Support for multiple website scraping simultaneously.
Use multi-qa-mpnet-base-dot-v1 for better semantic search accuracy.
Integrate caching mechanisms to improve response times.
Enhance GPT-4 prompt tuning to further prevent hallucinations.

Contact & GitHub

Email: himasai197@gmail.com
 GitHub: If you find this project useful, please  star the repository!

Hope you like the project! Let me know if you have suggestions! 



            Please refer the pdf file for Technical architecture 
            please refer the flow chart for system design Credits to app.eraser.co for AI flowchart
            