from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 1. Load the environment variables from your .env file
load_dotenv()

# 2. Initialize the embeddings model (it will now automatically find OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=64
)

texts = [
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]
vectors = embeddings.embed_documents(texts)
print(vectors)
