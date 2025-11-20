from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

def load_documents(directory):
    # Load the PDF or txt documents from the directory
    loader = DirectoryLoader(directory)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs

def main():
    # Get the documents split into chunks
    docs = load_documents(rag_directory)

    # Create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the documents into Chroma and save it to the disk
    Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")


if __name__ == "__main__":
    main()