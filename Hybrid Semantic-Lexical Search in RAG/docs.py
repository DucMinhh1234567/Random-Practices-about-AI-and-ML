import requests
import zipfile
import io
import os

url = "https://github.com/gakudo-ai/open-datasets/raw/refs/heads/main/asia_documents.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("asia_data")

docs = []
doc_names = []
for file in os.listdir("asia_data"):
    if file.endswith(".txt"):
        with open(f"asia_data/{file}", "r", encoding="utf-8") as f:
            docs.append(f.read())
            doc_names.append(file)
 
print(f"Loaded {len(docs)} documents for the knowledge base.")