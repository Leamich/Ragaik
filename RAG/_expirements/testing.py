
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"


from datasets import load_dataset
from langchain.schema import Document
from itertools import islice
from ..domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever

streamed_dataset = load_dataset(
    "math-ai/AutoMathText",
    split="train",
    streaming=True
)

retriever =FaissAndBM25EnsembleRetriever()

for idx, example in enumerate(islice(streamed_dataset, 1)):
    text = example["text"]
    document = Document(page_content=text, metadata={"source": "AutoMathText", "id": idx})
    retriever.add(document)

test_queries = [
    "What is the derivative of x^2?",
    "Solve the integral of sin(x) dx",
    "Define a bijection between sets"
]

for query in test_queries:
    results = retriever.query(query)
    print(f"Query: {query}\nResults:")
    for res in results:
        print(res)
    print("-"*40)
