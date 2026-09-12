from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from openai_config import DEFAULT_MODEL


documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))

llm = OpenAI(model=DEFAULT_MODEL, temperature=0.1)
Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
index = VectorStoreIndex.from_documents([document])


query_engine = index.as_query_engine()

response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))