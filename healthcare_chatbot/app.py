import streamlit as st
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import faiss
from langchain.chains.question_answering import load_qa_chain
from src.helper import load_pdf, text_split
from langchain_community.docstore.in_memory import InMemoryDocstore



# Load environment variables
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Set up Streamlit interface
st.title("Healthcare Chatbot")

# Load and process the PDF data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Since text_chunks is already a list of Document objects, use it directly
# documents = text_chunks
documents = []
for i,t in enumerate(text_chunks):
    x = Document(page_content = t.page_content, metadata = {'source':f'Source {i}'})
    documents.append(x)

# Create embeddings using SentenceTransformer
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode([doc.page_content for doc in documents])

# print('EMBEDDINGS: ',embeddings.shape)

# Create or load the FAISS index
index_path = "healthcare/index.faiss"
# if os.path.exists(index_path):
#     faiss_index = faiss.read_index(index_path)
# else:
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
# faiss_index.add(embeddings)
# faiss.write_index(faiss_index, index_path)

# Ensure the FAISS index and documents are aligned
# if faiss_index.ntotal != len(documents):
#     raise ValueError("The number of documents does not match the number of vectors in the FAISS index.")

# Initialize the FAISS vectorstore with the embedding function
vectorstore = FAISS(
    embedding_function=embedding_model.encode,  # Pass the embedding function here
    index=faiss_index,
    docstore = InMemoryDocstore(),
    index_to_docstore_id = {}
    # documents=documents  # Ensure this is passed as the correct parameter
)

ids = [str(i) for i in range(len(documents))]
vectorstore.add_documents(documents=documents, ids=ids)

# exit()
# Initialize the language model from Hugging Face
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # Replace with the appropriate model repo ID
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Define the QA chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

# Set up the RetrievalQA chain
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectorstore.as_retriever())

# User input
user_query = st.text_input("Enter your question:")

if user_query:
    # Get the response from the chatbot
    response = qa.run({"query": user_query})
    st.write(f"Response: {response}")
