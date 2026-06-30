from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

text = TextLoader("data/ndcs_kb.txt")
documents = text.load()
split = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=[
        "\n================",
        "\n--------",
        "\n--- ",
        "\n  RULE",
        "\n  IF",
        "\n\n",
        "\n"
    ]
)
splitted_doc = split.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    documents=splitted_doc, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)   
print("Database built successfully!")