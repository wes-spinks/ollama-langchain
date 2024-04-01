import sys
import ollama

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

messages = []


class ChatWebDoc:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral:instruct")
        # Loading embedding
        self.embedding = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')

        self.text_splitter = CharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks.
            You are an expert on Gartner market industries and products in those markets.
            Use only the following pieces of market context to answer the question.
            If you don't know the answer, say you don't know.[/INST] </s>
            [INST] Question: {input}
            Context: {context}
            Answer: [/INST]
            """
        )

    def ingest(self, csvdata):
        # Load web pages
        chunks = self.text_splitter.split_documents(csvdata)

        # Create vector store
        vector_store = SKLearnVectorStore.from_documents(
            documents=chunks, embedding=self.embedding, persist_path="./sklearn"
        )
        vector_store.persist()

    def load(self):
        # Load vector store
        vector_store = SKLearnVectorStore(
            embedding=self.embedding, persist_path="./sklearn"
        )

        # Create chain
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 1,
                "score_threshold": 0.4,
            },
        )

        document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, document_chain)

    def ask(self, query: str):
        if not self.chain:
            self.load()

        result = self.chain.invoke({"input": query})

        print(result["answer"])
        # for doc in result["context"]:
        #     print("Source: ", doc.metadata["source"])


def read_ingest_data(filepath="markets-and-products.csv"):
    loader = UnstructuredCSVLoader(
    file_path=filepath, mode="elements")
    docs = loader.load()
    return docs


def build():
    w = ChatWebDoc()
    src_csv = read_ingest_data()
    w.ingest(src_csv)


def chat():
    w = ChatWebDoc()

    w.load()

    while True:
        query = input(">>> ")

        if len(query) == 0:
            continue

        if query == "/exit":
            break

        w.ask(query)


if len(sys.argv) < 2:
    chat()
elif sys.argv[1] == "--ingest":
    build()


def send(chat):
    messages.append(
        {
            "role": "user",
            "content": chat,
        }
    )
    stream = ollama.chat(
        model="mistral:instruct",
        messages=messages,
        stream=True,
    )

    response = ""
    for chunk in stream:
        part = chunk["message"]["content"]
        print(part, end="", flush=True)
        response = response + part

    messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    print("")


while True:
    chat = input(">>> ")

    if chat == "/exit":
        break
    elif len(chat) > 0:
        send(chat)
