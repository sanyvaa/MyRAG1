from langchain_ollama import OllamaEmbeddings
#from dotenv import load_dotenv
from get_models import get_embedding_model_name


#from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    #load_dotenv()
    #embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    #)
    embeddings = OllamaEmbeddings(model=get_embedding_model_name())
    return embeddings
