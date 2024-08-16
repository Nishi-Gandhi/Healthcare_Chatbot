from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initialize FAISS and create a new index
index_name = "healthcare"
docsearch = FAISS.from_texts([t.page_content for t in text_chunks], embeddings)

# Save the index locally
docsearch.save_local(index_name)


