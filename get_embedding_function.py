from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

#from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    load_dotenv()
    #embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    #)
    embeddings = OllamaEmbeddings(model = str(os.getenv("CHAT_MODEL")))
    return embeddings
