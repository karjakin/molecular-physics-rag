import warnings
from llama_index.packs.raptor import RaptorRetriever, RaptorPack
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
    r"curso\unidad 1\4 calor y capacidad calorífica.pdf"
]).load_data()

client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")
vector_store = ChromaVectorStore(chroma_collection=collection)


raptor_pack = RaptorRetriever(
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

def call_doc(question):
    clear=[]
    nodes = raptor_pack.retrieve(question, mode="tree_traversal")
    for n in range(len(nodes)):
        clear.append(nodes[n].text)
    return clear

"""
question="que es un mol?"
a=call_doc(question)
print(a)
for element in a:
    print("--------------------")
    print(element)  


"""



