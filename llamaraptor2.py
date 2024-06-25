import warnings
from llama_index.packs.raptor import RaptorRetriever,RaptorPack
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import nest_asyncio
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
import httpx

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
nest_asyncio.apply()

documents = SimpleDirectoryReader(input_files=[
    r"curso\unidad 1\1 que es un mol.pdf",
    r"curso\unidad 1\2 límite termodinámico.pdf",
    r"curso\unidad 1\3 ecuación del gas ideal (empírica).pdf",
    r"curso\unidad 1\4 calor y capacidad calorífica.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\1 distrubuciones de probabilidad discretas.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\2 distrubuciones de probabilidad continuas.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\3 transformaciones lineales.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\4 varianza.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\5 variables independientes.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\6 distribución binomial.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\7 equilibrio térmico.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\8 termómetros.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\9 microestados y macroestados.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\10 definición estadística de temperatura.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\11 ensambles.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 2\12 aplicaciones de la distrubución de Boltzmann.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 3\1 distribución de maxwell-boltzmann.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 3\2 distribuciones moleculares.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 3\3 Ley del gas ideal.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 3\4 Ley de Dalton.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 4\1 efusión molecular.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 4\2 Recorrido libre medio y colisiones.pdf",
    r"C:\Users\JAIR\Documents\rag\curso\unidad 4\3 Propiedades de transporte.pdf"
]).load_data()

client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")
vector_store = ChromaVectorStore(chroma_collection=collection)

try:
    retriever = RaptorRetriever(
        documents,
        embed_model=OptimumEmbedding(folder_name="./bge_onnx"),
        llm=Ollama(model="llama3:70b-instruct", request_timeout=300.0, temperature=0.1),
        vector_store=vector_store,
        similarity_top_k=2,
        mode="collapsed",
        transformations=[
            SentenceSplitter(chunk_size=400, chunk_overlap=50)
        ],
    )
except httpx.ReadTimeout:
    print("Request timed out. Please try again later.")

llm = Ollama(model="llama3:70b-instruct", request_timeout=300.0, temperature=0.1)
synth = get_response_synthesizer(streaming=True, llm=llm)

def model_res_generator(query):
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synth
    )
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
    query = "¿que son las distribucciones moleculares?"
    for response_chunk in model_res_generator(query):
        print(response_chunk)
