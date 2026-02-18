import json

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 300

converter = DocumentConverter()

result = converter.convert("./2408.09869v5.pdf")
document = result.document

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    max_tokens=MAX_TOKENS,
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunks = list(chunker.chunk(document))

# Initialize the embedding model
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_dim = embedding_model.get_sentence_embedding_dimension()

paper_title = "N/A"
paper_url = "N/A"

with open("./test_output/docling_paper_metadata.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        for extraction in doc.get("extractions", []):
            extraction_class = extraction.get("extraction_class")
            extraction_text = extraction.get("extraction_text")

            if extraction_class == "title" and paper_title == "N/A":
                paper_title = extraction_text

            if extraction_class == "url" and paper_url == "N/A":
                paper_url = extraction_text

metadata_document_info = {
    "title": paper_title,
    "url": paper_url,
}

qdrant = QdrantClient(path="db/data")
qdrant.create_collection(
    collection_name="docling_paper",
    vectors_config=models.VectorParams(
        size=embedding_dim,
        distance=models.Distance.COSINE,
    ),
)

payload = []
embed = []
ids = []

for idx, chunk in enumerate(chunks):
    payload.append({"text": chunk.text, "metadata": metadata_document_info})
    # Generate embeddings using sentence-transformers
    chunk_embedding = embedding_model.encode(chunk.text)
    embed.append(chunk_embedding)
    ids.append(idx)

qdrant.upload_collection(
    collection_name="docling_paper",
    vectors=embed,
    ids=ids,
    payload=payload,
)

search_result = qdrant.query_points(
    collection_name="docling_paper",
    query=embedding_model.encode("what is docling?"),
).points

print(search_result[0].payload)
print(search_result[0].payload["text"])
print(search_result[0].payload["metadata"]["url"])