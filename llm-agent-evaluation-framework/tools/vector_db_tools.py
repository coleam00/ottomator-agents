import streamlit as st
import hashlib
import re

from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

@st.cache_resource
def get_chroma_instance():
    # Create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Get the Chroma instance from what is saved to the disk
    return Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

db = get_chroma_instance()

def string_to_vector_id(input_string: str, max_length: int = 64) -> str:
    """
    Converts a string into a vector-friendly ID by removing special characters, 
    replacing spaces with underscores, and optionally hashing the string if it exceeds max length.
    
    Arguments:
    - input_string (str): The input string to convert to a vector ID.
    - max_length (int, optional): The maximum length of the vector ID. Defaults to 64 characters.
    
    Returns:
    - str: A string that can be used as a vector ID.
    
    Example usage:
    string_to_vector_id("Example String For Vector ID")
    """
    # Remove non-alphanumeric characters (except spaces and underscores)
    sanitized_string = re.sub(r'[^a-zA-Z0-9\s_]', '', input_string)
    
    # Replace spaces with underscores
    sanitized_string = sanitized_string.replace(" ", "_")
    
    # Truncate if necessary
    if len(sanitized_string) > max_length:
        # If the string is too long, hash it to fit within the max length
        hash_object = hashlib.sha256(sanitized_string.encode())
        sanitized_string = hash_object.hexdigest()[:max_length]
    
    return sanitized_string

@tool
def query_documents(question: str) -> str:
    """
    Uses RAG to query documents for information to answer a question
    that requires specific context that could be found in documents

    Example call:

    query_documents("What are the action items from the meeting on the 20th?")
    Args:
        question (str): The question the user asked that might be answerable from the searchable documents
    Returns:
        str: The list of texts (and their sources) that matched with the question the closest using RAG
    """
    try:
        similar_docs = db.similarity_search(question, k=3)
        docs_formatted = list(map(lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", similar_docs))

        return str(docs_formatted)     
    except Exception as e:
        return f"Error querying the vector DB: {e}"

@tool
def add_doc_to_knowledgebase(file_path: str) -> str:
    """
    Adds a local document to the vector DB knowledgbase for RAG.
    This function can only be called on local documents - Google Drive docs must be downloaded first.
    The content of the file is put in the vector DB with the metadata
    including the file source. ID is randomly generated.

    Example call:

    add_doc_to_knowledgebase("/path/to/local/file")
    Args:
        file_path (str): The local path to the file to add to the knowledgebase (NOT Google Drive)
    Returns:
        str: The success of the operation of adding the document to the vector DB
    """
    try:
        loader = TextLoader(file_path)
        doc_arr = loader.load()
        db.add_documents(documents=doc_arr, ids=[string_to_vector_id(file_path.split("/")[-1])])
        return "Successfully added the file to the knowledgebase."
    except Exception as e:
        return f"Error adding file to knowledgbase: {e}"

@tool
def clear_knowledgebase() -> str:
    """
    Removes all documents from the vector DB knowledgebase to clear it.

    Example call:

    clear_knowledgebase()
    Returns:
        str: The success of the operation of clearing the vector DB
    """
    try:
        db.reset_collection()
        return "Successfully cleared the knowledgebase."
    except Exception as e:
        return f"Error clearing the knowledgbase: {e}"


# Maps the function names to the actual function object in the script
# This mapping will also be used to create the list of tools to bind to the agent
available_vector_db_functions = {
    "query_documents": query_documents,
    "add_doc_to_knowledgebase": add_doc_to_knowledgebase,
    "clear_knowledgebase": clear_knowledgebase
}      
