import os
import streamlit as st
import pickle
import time
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()  # take environment variables from .env (especially openai api key)

# Streamlit page configuration
st.set_page_config(page_title="RockyBot: News Research Tool", page_icon="📈")

# Custom styles for the app
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    .stApp {
        padding: 20px;
    }
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1558b3;
    }
    .stTextInput > div > div > input {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a73e8;
    }
    .stSidebar {
        background-color: #fff;
        border-right: 1px solid #ddd;
        padding: 20px;
    }
    .stSidebar > div > div {
        color: #333;
    }
    .stTitle {
        color: #1a73e8;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .stSubtitle {
        color: #444;
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #1a73e8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and sidebar title
st.title("InfoBot: Intelligent Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")

# Initialize session state
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'urls' not in st.session_state:
    st.session_state.urls = []

# Input fields for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_index_file = "faiss_index"

main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.9, max_tokens=500)

# Function to extract text from URL
def extract_text_from_url(url):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    start_time = time.time()
    downloaded_size = 0
    content = ''
    for data in response.iter_content(1024):
        downloaded_size += len(data)
        content += data.decode('utf-8', errors='ignore')
        # elapsed_time = time.time() - start_time
        # download_speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
        # percent_done = (downloaded_size / total_size) * 100 if total_size > 0 else 0
        # print(f"Loading {url} - {percent_done:.2f}% complete at {download_speed:.2f} bytes/second")
    soup = BeautifulSoup(content, 'html.parser')
    return ' '.join(p.get_text() for p in soup.find_all('p'))

# Process URLs if the button is clicked and URLs are non-empty
if process_url_clicked and any(urls):
    print("Process URLs button clicked and URLs are non-empty.")
    print(f"URLs provided: {urls}")
    
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = []
        for i, url in enumerate(urls):
            if url:
                print(f"Loading URL {i+1} out of {len(urls)}: {url}")
                text = extract_text_from_url(url)
                data.append({"content": text, "source": url})
                print(f"Loaded URL {i+1} successfully.")
        print("Data successfully loaded for all URLs.")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Split data
    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    print("Text splitter initialized.")
    
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    try:
        print("Attempting to split documents...")
        docs = []
        for doc in data:
            doc_splits = text_splitter.split_text(doc['content'])
            for i, split in enumerate(doc_splits):
                docs.append({"page_content": split, "source": doc['source']})
                print(f"Split {i+1} for document from {doc['source']}")
        print("Documents successfully split.")
        
        print(f"Number of documents: {len(docs)}")
        for i, doc in enumerate(docs[:5]):
           print(f"Document {i+1} snippet: {doc}...")  # Show a snippet of the first few documents
        
        # print(type(docs))
        # for doc in docs: print(doc.keys())
        # print(f"Data structure of docs: {docs[0]}")
    except Exception as e:
        print(f"Error splitting documents: {e}")

    # Convert to Document objects
    print("Converting to Document objects...")
    docs = [Document(page_content=doc['page_content'], metadata={"source": doc['source']}) for doc in docs]
    print("Conversion complete.")

    # Create embeddings and save to FAISS index
    print("Initializing embeddings...")
    embeddings = OpenAIEmbeddings()
    print("Embeddings initialized.")
    
    try:
        print("Creating FAISS index from documents...")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        print("Embedding vector successfully built.")
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        print("FAISS index created.")
    except Exception as e:
        print(f"Error creating embedding vector: {e}")
    
    time.sleep(2)

    # Save the FAISS index
    try:
        print("Attempting to save FAISS index...")
        vectorstore_openai.save_local(faiss_index_file)
        print("FAISS index successfully saved.")
        
        # Update session state
        st.session_state.faiss_index = vectorstore_openai
        st.session_state.embeddings = embeddings
        st.session_state.urls = urls
        
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

# Input for query
query = st.text_input("Question: ", key="query")
submit_query = st.button("Submit Query")

# Process query
if submit_query and query:
    print(f"Query received: {query}")
    if st.session_state.faiss_index:
        print("FAISS index found in session state.")
        try:
            print("Initializing RetrievalQAWithSourcesChain...")
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.faiss_index.as_retriever())
            print("RetrievalQAWithSourcesChain initialized.")
                
            print(f"Processing query: {query}")
            result = chain({"question": query}, return_only_outputs=True)
            print("Query processed.")
                
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            print("Answer:")
            print(result["answer"])
                
            # Display answer on the screen
            st.write("**Answer:**")
            st.write(result["answer"])
                
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                print("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                st.write("**Sources:**")
                for source in sources_list:
                    print(source)
                    st.write(source)
            else:
                print("No sources found.")
                st.write("No sources found.")
        except Exception as e:
            print(f"Error processing query: {e}")
            st.write(f"Error processing query: {e}")
    else:
        print("FAISS index not found in session state. Please process URLs first.")
        st.write("FAISS index not found in session state. Please process URLs first.")
else:
    print("No query entered or submit button not clicked.")
