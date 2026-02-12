# Prerequisites:
    VSCode
    Python 3.13

# Frameworks and Libs:
    Langchain
    ChromaDB    
    Streamlit

# LLM providers    
    Ollama
    llama.cpp
    openai
    google_genai
    anthropic
    mistralai

# LLM's
    # embeddings LLM
    mxbai-embed-large 
    
    #Chat model
    llama3.2

# Setup:
    # Create virtual environment
    https://code.visualstudio.com/docs/python/python-tutorial

    # add uv package installer (more effictive than pip)    
    pip install uv

    #install packages from pyproject.toml
    uv sync

# Ollama
    install Ollama
    pull Ollama LLM's
    run Ollama

# chroma folder contains vector database that includes one TestSuite documentation bundle 020_Landmark_TestLine_MPE

# Populating vector database
    # copy documentation folder with htm (html) files to some local folder, by default it is 
    # 'C:/MTS TestSuite Documentation/020_Landmark_TestLine_MPE'
    # change "Populate RAG DB.py" to point to this folder
    # run Ollama, it will run embeddings LLM
    # run "Populate RAG DB.py" 
# NOTE: it could take up to 1 hour to parse files in one TestSuite documentation bundle and to create  embeddings in vector DB


# AI Chat
    # run Ollama
    # open terminal, wait for venv is activated
    # run "streamlit run .\AIChat.py" command, it should open web browser with chat window

# NOTE: to use public LLM's (from OpenAI, Google etc...) add '.evn' file and specify your personal API keys for these models
    