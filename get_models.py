from langchain_ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_model_name():
    embeddings = "mxbai-embed-large"
#    embeddings = "qwen3-embedding"
    return embeddings

def get_main_model_name():
    main = "llama3.2"
    #main = "qwen3-vl:4b"
    #main = "deepseek-r1:1.5b"
    return main