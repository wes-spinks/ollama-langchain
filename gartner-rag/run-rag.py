from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import sys
import ollama

messages = []
 
class ChatWebDoc:
    vector_store = None
    retriever = None
    chain = None
 
    def __init__(self):
        self.model = ChatOllama(model="mistral:instruct")
        #Loading embedding
        self.embedding = FastEmbedEmbeddings()
 
        self.text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use only the following pieces of retrieved con            text to answer the question. If you don't know the answer, just say that you don't know. [/INST] </s> 
            [INST] Question: {input} 
            Context: {context} 
            Answer: [/INST]
            """
        )
 
    def ingest(self, url_list):
        #Load web pages
        docs = WebBaseLoader(url_list).load()
        chunks = self.text_splitter.split_documents(docs)
 
        #Create vector store
        vector_store = Chroma.from_documents(documents=chunks, 
            embedding=self.embedding, persist_directory="./chroma_db")
 
    def load(self):
        #Load vector store
        vector_store = Chroma(persist_directory="./chroma_db", 
            embedding_function=self.embedding)
 
        #Create chain
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
 
        document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, document_chain)
 
    def ask(self, query: str):
        if not self.chain:
            self.load()
 
        result = self.chain.invoke({"input": query})
 
        print(result["answer"])
        for doc in result["context"]:
            print("Source: ", doc.metadata["source"])

def read_ingest_urls(filename="gartner_markets_urls.txt"):
    with open(filename, "r") as pathsfile:
        return pathsfile.readlines() 
 
def build():
    w = ChatWebDoc()
    src_list = read_ingest_urls()
    w.ingest(src_list)
 
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
      'role': 'user',
      'content': chat,
    }
  )
  stream = ollama.chat(model='mistral:instruct', 
    messages=messages,
    stream=True,
  )
 
  response = ""
  for chunk in stream:
    part = chunk['message']['content']
    print(part, end='', flush=True)
    response = response + part
 
  messages.append(
    {
      'role': 'assistant',
      'content': response,
    }
  )
 
  print("")
 
while True:
    chat = input(">>> ")
 
    if chat == "/exit":
        break
    elif len(chat) > 0:
        send(chat)
