# Standard Library Imports
import openai, json, nltk, string, requests

# Third-Party Imports
import streamlit as st
import langchain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.llms import OpenAI

# Download necessary nltk resources
nltk.download('punkt')

# Function to load in data found in the 'data' folder of the central repository; To upload your own data, simply remove the existing data in that folder and upload your own. Don't forget to update the prompt below!
@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(text="Loading and indexing the data – hang tight! This shouldn't take more than a minute."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on the Innovation CoPilot and your job is to answer questions about it. Assume that all questions are related to the Innovation CoPilot. Keep your answers technical and based on facts – do not hallucinate features."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index