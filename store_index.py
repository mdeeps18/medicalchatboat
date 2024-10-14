from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve Pinecone API key and environment variables from .env
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Ensure API key and environment are correctly loaded
if not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("Pinecone API key or environment not set in .env file")

# Extract and process the PDF data
extracted_data = load_pdf("data/")  # Assuming PDFs are in 'data/' directory
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define the name of the Pinecone index
index_name = "boatchat"

# Check if the index exists, if not, create it
if index_name not in pinecone.list_indexes():
    # Get the embedding dimension
    embedding_dimension = len(embeddings.embed_query("test"))
    print(f"Creating index '{index_name}' with dimension {embedding_dimension}...")
    pinecone.create_index(index_name, dimension=embedding_dimension)
else:
    print(f"Index '{index_name}' already exists. Skipping index creation.")

# Connect to the existing index
print(f"Connecting to the existing index '{index_name}'...")
index = pinecone.Index(index_name)

# Create embeddings for the text chunks and store them in Pinecone
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
