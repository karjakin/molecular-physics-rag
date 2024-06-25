from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core import Settings

OptimumEmbedding.create_and_save_optimum_model(
    "BAAI/bge-m3", "./bge_onnx"
)

Settings.embed_model = OptimumEmbedding(folder_name="./bge_onnx")
