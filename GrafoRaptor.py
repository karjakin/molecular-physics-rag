from typing import  TypedDict
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import Field
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings,GPT4AllEmbeddings
import pprint
from langsmith import traceable
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads
from operator import itemgetter

import tiktoken
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from langchain.text_splitter import RecursiveCharacterTextSplitter


#Verificar hay Jaques
##Hay capturas de piezas o intercambio de piezas
#Hay amenazas de mate o jaques
#Cual sera la estrategia a 5 jugadas


#fen
#tema principal pregunata
#Teorica o practica

RANDOM_SEED = 224  # Fixed seed for reproducibility

# Ollama model name
local_llm = "llama3:70b-instruct"
llm = ChatOllama(model=local_llm, temperature=0)








prompt_query = PromptTemplate(
    template="""Get the most iconic chapter or part to illustrate based on the book petition \n
    Here is the user book petition: '{petition}' \n
    Formulate a question that is well optimized for retrieval. \n
    """,
    input_variables=["petition"],
)

prompt_pregunta_chess = PromptTemplate(
    template="""You are a helpful assistant that generates a question related to chess. \n
    Formulate a question that is well optimized for retrieval. \n
    Generate a question related to chess based on the question: \n
    {input} \n
    Just return the question. \n
    """,
    input_variables=["input"],
)

prompt_query_pregunta_inicial = PromptTemplate(
    template="""
    Based on the chapter '{chapter}' from the book '{book}' create a question that extracts all the relevant information,facts and details from the chapter. \n
    Formulate a question that is well optimized for retrieval. \n
    """,
    input_variables=["book", "chapter"],
)


prompt_query_chapter_part = PromptTemplate(    
    template="""
    Based on the book '{book}', make a retival question that extracts the specific information from part '{part}' of the chapter '{chapter}' \n
    The question must include the specific part of the chapter. \n
    """,
    input_variables=["book", "chapter", "part"],
)

prompt_set_question = PromptTemplate(
    template=
    """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
    Each question must be different but related to the must important parts of the chapter {chapter} from the book {book}. \n
    Generate multiple search queries related to the question {question} \n
    Output (3 queries) separated by a new line. \n
    """,
    input_variables=["question", "book", "chapter"],
)

prompt_set_question_chess = PromptTemplate(
    template=
    """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
    Each question must be different but related to the must important things about the theme and the question. \n
    The main purpouse is to generate a correct answer and extract the most relevant information related to the question. \n
    The question must always be related to chess. \n
    Generate multiple search queries related to:
    \n  Question: {input} \n
    \n  Theme: Chess \n
    Output (3 queries) separated by a new line. \n
    """,
    input_variables=["input"],
)

prompt_final_answer_chess = PromptTemplate(
    template=
    """Your job is to create an answer that is fluid based on the chess related question and the final relevant retrived information. \n:
    \n  Question: {question} \n
    
    \n  Information: {info} \n
    """,
    input_variables=["question", "info"],
)

prompt_template_question_recursive = PromptTemplate(
    template="""Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """,
    input_variables=["question", "q_a_pairs", "context"],
)



llm = ChatOllama(model=local_llm, temperature=0)    


# Local embedding model paths (downloaded above)O
embd_model_path = r'C:\Users\JAIR\Documents\Paper chesscluster\nomic-embed-text-v1.5-GGUF\nomic-embed-text-v1.5.f16.gguf'
embedding = LlamaCppEmbeddings(model_path=embd_model_path, n_batch=512)

###


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


### --- Our code below --- ###


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = embedding.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """Here is a sub-set of LangChain Expression Langauge doc. 
    
    LangChain Expression Langauge provides a way to compose chain in LangChain.
    
    Give a detailed summary of the documentation provided.
    
    Documentation:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # Format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Load text files RAPTOR------------------------------------------------------------------------------------------------------------------------------------------------
file_paths = [
    "curso//unidad 1//1 que es un mol.pdf",
    "curso//unidad 1//2 límite termodinámico.pdf",
    "curso//unidad 1//3 ecuación del gas ideal (empírica).pdf",
    "curso//unidad 1//4 calor y capacidad calorífica.pdf"
]

from PyPDF2 import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    content = ""
    for page in reader.pages:
        content += page.extract_text()
    return content


docs = []
for file_path in file_paths:
    content = read_pdf(file_path)
    doc = {"page_content": content, "metadata": {"source": file_path}}
    docs.append(doc)

docs_texts = [d["page_content"] for d in docs]

# Calculate the number of tokens for each document
counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]





# Doc texts concat
d_sorted = sorted(docs, key=lambda x: x["metadata"]["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc["page_content"] for doc in d_reversed]
)

print(
    "Num tokens in all context: %s" % num_tokens_from_string(concatenated_content, "cl100k_base")
)

# Split the concatenated content into chunks
chunk_size_tok = 512
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok, chunk_overlap=0
)
texts_split = text_splitter.split_text(concatenated_content)
print("Chunks:", len(texts_split))





# Build tree
leaf_texts = docs_texts
results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)


print("Results------------------")
# Initialize all_texts with leaf_texts
all_texts = leaf_texts.copy()

# Iterate through the results to extract summaries from each level and add them to all_texts
for level in sorted(results.keys()):
    # Extract summaries from the current level's DataFrame
    summaries = results[level][1]["summaries"].tolist()
    # Extend all_texts with the summaries from the current level
    all_texts.extend(summaries)


# Index
vectorstore = Chroma.from_texts(
    texts=all_texts,
    collection_name="rag-chroma-raptor",
    embedding=embedding,
)
retriever = vectorstore.as_retriever()

###

#chains

chain_input = prompt_query | llm | StrOutputParser()
chain_input_pregunta_chess = prompt_pregunta_chess | llm | StrOutputParser()
chain_input_pregunta_inicial = prompt_query_pregunta_inicial | llm | StrOutputParser() 
chain_set_question = prompt_set_question | llm | StrOutputParser()
chain_set_question_chess =prompt_set_question_chess | llm | StrOutputParser()
chain_final_ans_chess = prompt_final_answer_chess | llm | StrOutputParser()
chain_input_chapter_part = prompt_query_chapter_part | llm | StrOutputParser()
chain_q_recursive = prompt_template_question_recursive | llm | StrOutputParser()





class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]



### Nodes ###


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = PromptTemplate(
        template="""Make a final answer for the user, use the information retrieved to construct your final answer. \n
        Just return the answer. \n
        Use less than 1000 tokens. \n
        Here is the user question: '{question}' \n
        Here is the context : '{context}' \n
        
        """,
        input_variables=["question", "context"],
    )

    # LLM
    llm = ChatOllama(model=local_llm, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user petition. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user petition: {question} \n
        If the document contains information related to the petition, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["question","context"],
    )

    # Chain
    chain = prompt | llm | JsonOutputParser()

    # Score
    filtered_docs = []
    for d in documents:
        score = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
            }
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"keys": {"documents": filtered_docs, "question": question}}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # LLM
    llm = ChatOllama(model=local_llm, temperature=0)
    
    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question:""",
        input_variables=["question"],
    )

    # Chain
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question}}

def prepare_for_final_grade(state):
    """
    Passthrough state for final grade.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): The current graph state
    """

    print("---FINAL GRADE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]

    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


### Edges ###

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents(state):
    """
    Determines whether the generation is grounded in the document.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs DOCUMENTS---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]

    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer contains a chapter script related to the question {question} \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' score to indicate whether the answer is good by a set of facts. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "documents"],
    )

    # Chain
    chain = prompt | llm | JsonOutputParser()
    score = chain.invoke({"generation": generation, "documents": documents, "question": question})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: SUPPORTED, MOVE TO FINAL GRADE---")
        return "supported"
    else:
        print("---DECISION: NOT SUPPORTED, GENERATE AGAIN---")
        return "not supported"


def grade_generation_v_question(state):
    """
    Determines whether the generation addresses the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---GRADE GENERATION vs QUESTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    generation = state_dict["generation"]

    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether an answer is useful to resolve the petition. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the petition: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a petition. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.""",
        input_variables=["generation", "question"],
    )

    # Prompt
    chain = prompt | llm | JsonOutputParser()
    score = chain.invoke({"generation": generation, "question": question})
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: USEFUL---")
        return "useful"
    else:
        print("---DECISION: NOT USEFUL---")
        return "not useful"


# ## Build Graph


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("prepare_for_final_grade", prepare_for_final_grade)  # passthrough

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents,
    {
        "supported": "prepare_for_final_grade",
        "not supported": "generate",
    },
)
workflow.add_conditional_edges(
    "prepare_for_final_grade",
    grade_generation_v_question,
    {
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

def extract_questions(data:str):
    # Extract the str questions from the data  per sentence separated by \n and return a list of questions
    questions = data.split("\n")
    return questions

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()


def generate_question_set(question,book,chapter):
    # Run
    question_input =  chain_set_question.invoke({"question":question, "book":book, "chapter":chapter})
    return question_input

def generate_question_set_chess(input):
    # Run
    question_input =  chain_set_question_chess.invoke({"input":input})
    return question_input

def generate_question(petition,book,chapter):
    # Run
    question_input =  chain_input.invoke({"petition":petition, "book":book, "chapter":chapter})
    inputs = {"keys": {"question": question_input}}
    return inputs

def generate_question_chess(input):
    # Run
    question_input =  chain_input_pregunta_chess.invoke({"input":input})
    inputs = {"keys": {"question": question_input}}
    return inputs

def generate_question_inicial(book, chapter):
    # Run
    question_input =  chain_input_pregunta_inicial.invoke({"book":book, "chapter":chapter})
    inputs = {"keys": {"question": question_input}}
    return inputs

def generate_final_answer(question,q_a_pairs,context):
    # Run
    answer =  chain_q_recursive.invoke({"question":question, "q_a_pairs":q_a_pairs, "context":context})
    return answer

def generate_final_ans_chess(question,list_ans):
    ans = chain_final_ans_chess.invoke({"question":question, "info":list_ans})
    return ans

def generate_question_part(book, chapter, part):
    # Run
    question_input =  chain_input_chapter_part.invoke({"book":book, "chapter":chapter, "part":part})
    inputs = {"keys": {"question": question_input}}
    return inputs

def set_preguntas(question,book,chapter):
    capitulos = []
    lista_respuestas = []
    pregunta_set = generate_question_set(question,book,chapter)
    print(pregunta_set)
    preguntas = extract_questions(pregunta_set)

    i = -1
    for pregunta in preguntas:
        inputs = {"keys": {"question": pregunta}}
        res = run_workflow(inputs)
        lista_respuestas.append(format_qa_pair(pregunta,res))
        if i == -1:
            respuesta = generate_final_answer(pregunta,"No previous answer",res)     
        else:
            respuesta = generate_final_answer(pregunta,lista_respuestas[i],res)
        capitulos.append(respuesta)
        i += 1



    return capitulos


def set_preguntas_chess(input):

    lista_respuestas = []
    pregunta_set = generate_question_set_chess(input)
    print(pregunta_set)
    preguntas = extract_questions(pregunta_set)
    res_str = ""
    i = -1
    for pregunta in preguntas:
        inputs = {"keys": {"question": pregunta}}
        res = run_workflow(inputs)
        lista_respuestas.append(format_qa_pair(pregunta,res))
        if i == -1:
            respuesta = generate_final_answer(pregunta,"No previous answer",res)     
        else:
            respuesta = generate_final_answer(pregunta,lista_respuestas[i],res)

        res_str = res_str + "\n" + respuesta
        i += 1

    response = generate_final_ans_chess(pregunta_set,res_str)

    return response

def run_workflow(inputs):
    for output in app.stream(inputs, {"recursion_limit": 50}):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            #pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value['keys']['generation'])
    return value['keys']['generation']




import csv

def process_csv(file_path):
    # Open the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Process each row
    for i in range(1, len(rows)):  # Start from index 1 to skip the header row
        row = rows[i]
        input_data = row[0]  # Assuming the input data is in the first column (index 0)
        input = generate_question_chess(input_data)
        # Call the run_workflow function with the input data
        answer = run_workflow(input)

        # Update the "b" column with the answer
        row[1] = answer  # Assuming the "b" column is the second column (index 1)

  # Write the updated data back to the CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

question=run_workflow(generate_question_chess("que es un mol?"))
print(question) 
# Example usage
#file_path = 'data.csv'
#process_csv(file_path)