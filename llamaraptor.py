from llama_index.packs.raptor import RaptorPack, RaptorRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import nest_asyncio
from llama_index.core import Settings
Settings.embed_model = OptimumEmbedding(folder_name="./bge_onnx")

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=[r"curso\unidad 1\1 que es un mol.pdf",r"curso\unidad 1\2 límite termodinámico.pdf",r"curso\unidad 1\3 ecuación del gas ideal (empírica).pdf",r"curso\unidad 1\4 calor y capacidad calorífica.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")

vector_store = ChromaVectorStore(chroma_collection=collection)



"""
retriever = RaptorPack(
    documents,
    embed_model=OptimumEmbedding(folder_name="./bge_onnx"),
    llm = Ollama(model="llama3:70b-instruct", request_timeout=60.0,temperature=0.1),
    vector_store=vector_store,  
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ], 
)
"""


retriever = RaptorRetriever(
    documents,
    embed_model=OptimumEmbedding(folder_name="./bge_onnx"),
    llm = Ollama(model="llama3:70b-instruct", request_timeout=60.0,temperature=0.1),
    vector_store=vector_store,  
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ], 
)

from llama_index.core.query_engine import RetrieverQueryEngine


"""
query_engine = RetrieverQueryEngine.from_args(
    retriever,  llm = Ollama(model="llama3:70b-instruct", request_timeout=60.0,temperature=0.1)
)

response = query_engine.query("¿como se obtiene la capacidad calorifica?, ejemplo")
print(str(response))
print("-----------------------------------")
response = query_engine.query("¿cual es la ecuación del gas ideal?, muestra ejemplos")
print(str(response))


class ConsoleApp:
    def __init__(self):
        self.retriever = retriever  # Initialize your retriever here
        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever, llm=Ollama(model="llama3:70b-instruct", request_timeout=60.0, temperature=0.1)
        )

    def run(self):
        while True:
            query = input("Please enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            response = self.query_engine.query(query)
            print(str(response))
            print("-----------------------------------")

if __name__ == "__main__":
    app = ConsoleApp()
    app.run()
"""
from llama_index.core import get_response_synthesizer

llm = Ollama(model="llama3:70b-instruct", request_timeout=60.0, temperature=0.1)
synth = get_response_synthesizer(streaming=True, llm=llm)


"""
class ConsoleApp:
    def __init__(self):
        self.retriever = retriever
        self.query_engine = query_engine

    def run(self):
        while True:
            query = input("Please enter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            streaming_response = self.query_engine.query(query)
            print("Response: ")
            streaming_response.print_response_stream()
            print("-----------------------------------")

if __name__ == "__main__":
    app = ConsoleApp()
    app.run()
    
"""

def model_res_generator(query):
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synth
    )
    # Initiating a streaming query using the query engine
    streaming_response = query_engine.query(query)
    for text in streaming_response.response_gen:
        yield text

def run_app():
    while True:
        query = input("Please enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        print("Response: ")
        for response_chunk in model_res_generator(query):
            print(response_chunk)
        print("-----------------------------------")

if __name__ == "__main__":
    #run_app()
    query = "que es un mol?"
    for response_chunk in model_res_generator(query):
        print(response_chunk)