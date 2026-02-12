import os
import shutil
import re
from bs4 import BeautifulSoup
from pathlib import Path
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from pathlib import Path
import matplotlib.pyplot as plt

CHROMA_PATH = "chroma_with_images"

def main():
   documents = load_images()





def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    #soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def extract_image_sources(html_file_path):

#    embedding_function = OpenCLIPEmbeddingFunction()
    data_loader = ImageLoader()


#    client = chromadb.PersistentClient(path=CHROMA_PATH)
#    collection = client.create_collection(name='multimodal_collection',
#        embedding_function=embedding_function,  data_loader=data_loader)

    with open(html_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    image_sources = []
    for img_tag in soup.find_all('img'):
        src = img_tag.get('src')
        if src:
            image_sources.append(src)
    return image_sources

def load_images():

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


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
