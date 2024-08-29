from enum import Enum
#from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.messages import HumanMessage, AIMessage
#from langchain_community.llms.ollama import Ollama
#from langchain_openai import OpenAI

from langchain_community.vectorstores import Qdrant

#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers.string import StrOutputParser

from langchain.chat_models import ChatOpenAI

class EmbeddingTypes(Enum):
    OLLAMA=1
    SENTENCE_TRANSFORMERS=2
    OPENAI=3

#from langchain.callbacks.base import BaseCallbackHandler


class LLMApp:
    def __init__(self, texts):
        self.texts = texts
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25, length_function=len)

    def split_documents(self):
        docs = self.text_splitter.create_documents([self.texts])
        return docs
    
    def handle_streamed_response(stream):
        for chunk in stream:
            print(chunk, end="", flush=True)
    
    def llm_processor(self, retriever, chat_history, input_question):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        prompt_search_query = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("system", contextualize_q_system_prompt)
        ])

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, streaming=True) #Ollama(model="llama2:chat")

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

        

        prompt_get_answer = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based only on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

        #chat_history = [HumanMessage(content="What is a vector?"), AIMessage(content="A vector is a point in space.")]
        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": input_question
        })
        return response["answer"]


class EmbeddingModel:
    def __init__(self, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = model
        self.embedding_model = HuggingFaceEmbeddings(model_name=model)

    def generate_embedding(self, text):
        return self.embedding_model.embed_query(text)


class DatabaseModel:
    def __init__(self, embedding_model, collection_name: str, host: str="qdrant", port: int=6333):
        self.db_client = QdrantClient(host=host, port=port)
        self.embedding_model = embedding_model

        self.create_collection(collection_name=collection_name)
        
        self.qdrant = Qdrant(self.db_client, collection_name, embedding_model.embed_query)
    
    def create_collection(self, collection_name: str):
        if not self.db_client.collection_exists(collection_name=collection_name):
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

    def add_to_db(self, llm_app):
        docs = llm_app.split_documents()
        self.qdrant.add_documents(docs)

    def get_qdrant_as_retriever(self):
        return self.qdrant.as_retriever()
