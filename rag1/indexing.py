import os

from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path



load_dotenv()


embedding = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    check_embedding_ctx_length=False,
    chunk_size=16,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "pdfs" / "penguins.pdf"

chunks = extract_paragraphs(str(PDF_PATH))

vector_store = InMemoryVectorStore(embedding)
vector_store.add_texts(texts=chunks)

print(f"indexed {len(chunks)} chunks")


results = vector_store.similarity_search("Why are penguins good swimmers?", k=3)

for doc in results:
    print(doc.page_content)
    print("---")