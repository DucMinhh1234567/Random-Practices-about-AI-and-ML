from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path="./data",
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()

# print(docs)
# print(len(docs))

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

splits = text_splitter.split_documents(docs)

# from pprint import pprint
# pprint(splits)

embed = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

vector_store = FAISS.from_documents(
    documents=splits,
    embedding=embed,
    distance_strategy=DistanceStrategy.COSINE
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.2},
)

template = (
    "You are a strict, citation-focused assistant for a private knowledge base. \n"
    "RULES: \n"
    "1) Use ONLY the provided context to answer. \n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents. \"\n"
    "3) Do NOT use outside knowledge, guessing, or web information. \n"
    "4) If applicable, cite sources as (source:page) using the metadata. \n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

rag_chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = input("Question: ")

answer = rag_chain.invoke(question)

print(answer)