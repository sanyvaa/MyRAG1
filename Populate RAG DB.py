import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import BSHTMLLoader
from pathlib import Path
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from pathlib import Path
import matplotlib.pyplot as plt

CHROMA_PATH = "chroma_with_images"

def main():
  
    #clear_database()
    # Create (or update) the data store.
    documents = load_documents()
    #chunks = split_documents(doc)
    #add_to_chroma(chunks)





def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    #soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


    client = chromadb.PersistentClient(path=str("chroma_images"))
    data_loader = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()

    collection = client.get_or_create_collection(
         name='multimodal_collection',
         embedding_function=embedding_function,
         data_loader=data_loader)
    
   
    #uri1 = Path(r"C:\MTS TestSuite Documentation\020_Landmark_TestLine_MPE\Content\Source_TS_Masters\001_TS_Common\Graphics\Add_Channel_Resource.png").as_uri()
    uri1 = str(Path(r"d:\1.png"))
    collection.add(
        ids=["id1"],
        uris= [uri1],
        metadatas=[{"description":"dog"} ]
        )

    #results = collection.query(
    #    query_uris=[uri1] # A list of strings representing URIs to data
    #    )
    #results = collection.query(query_texts=["A picture of a dog"], n_results=1, include=['distances', 'data', 'uris'])

    query_results = collection.query(query_texts="A picture of a dog", n_results=2,  
                                     include=['data'])  

    data = query_results["data"]
    plt.imshow(data[0][0])
    plt.axis("off") 
    plt.show()
    return

    py_files = list(Path('C:/MTS TestSuite Documentation/020_Landmark_TestLine_MPE').rglob("Ac_MPE_MainSystem.htm"))
    docs = [Document]

    i=0
    for file in py_files:
        print(str(i) + " - " + str(file))
        
        try:
            image_sources = extract_image_sources(file)

            i += 1
        except Exception as e:
             print("cannot load " + str(file))               


    print("files loaded: " + str(len(docs)))
    return docs

def load_documents():
    
    py_files = list(Path('C:/MTS TestSuite Documentation/020_Landmark_TestLine_MPE').rglob("*.htm"))
    docs = [Document]
    print("number of htm files: " + str(len(py_files)))
    i=0
    for file in py_files:
        print(str(i) + " - " + str(file))
        
        try:
            document_loader = BSHTMLLoader(file_path=file)
            doc = document_loader.load()
            #soup = BeautifulSoup(doc.page_content, "lxml")
            #docs.append(doc)
            chunks = split_documents(doc)
            add_to_chroma(chunks)

            i += 1
        except Exception as e:
             print("cannot load " + str(file))               

#    document_loader = RecursiveUrlLoader(
#        "file:///C:/MTS%20TestSuite%20Documentation/020_Landmark_TestLine_MPE/Default.htm#Source_TS_Bundles/020_Landmark_TestLine_MPE/020_Bundle_ManualTitlePages/LM_MPE_Help_030_MP_SVH.htm", 
#         extractor = bs4_extractor )
    
    #document_loader = PyPDFDirectoryLoader(DATA_PATH)

    print("files loaded: " + str(len(docs)))
    return docs


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    #db.collection.delete(ids=[])
    
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
#        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
