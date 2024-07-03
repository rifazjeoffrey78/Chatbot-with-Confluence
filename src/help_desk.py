import sys

from langchain.memory.chat_memory import BaseChatMemory
from langchain_community.chat_models.premai import chat_with_retry

import load_db
import collections

from config import (EMBEDDING_MODEL)
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama

class HelpDesk():
    """Create the necessary objects to create a QARetrieval chain"""
    def __init__(self, new_db=True):
        self.new_db = new_db
        self.template = self.get_template()
        self.embeddings = self.get_embeddings()
        self.llm = self.get_llm()
        self.prompt = self.get_prompt()
        self.chatHistory = ChatMessageHistory()
        self.summary = "Initialize summary"

        if self.new_db:
            self.db = load_db.DataLoader().set_db(self.embeddings)
        else:
            self.db = load_db.DataLoader().get_db(self.embeddings)

        self.retriever = self.db.as_retriever()
        self.retrieval_qa_chain = self.get_retrieval_qa()


    def get_template(self):
        template = """
        Given this text extracts:
        -----
        {context}
        -----
        Please answer with to the following question:
        Question: {question}
        Helpful Answer:
        """
        return template

    def get_prompt(self) -> ChatPromptTemplate:
        # prompt = PromptTemplate(
        #     template=self.template,
        #     input_variables=["context", "question"]
        # )
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        return prompt

    def get_embeddings(self) -> OllamaEmbeddings:
        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        return embeddings

    def get_llm(self):
        llm = ChatTogether()
        #llm = ChatOllama(model="llama3")
        return llm

    def get_retrieval_qa(self):
        # chain_type_kwargs = {"prompt": self.prompt}
        # qa = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.retriever,
        #     return_source_documents=True,
        #     chain_type_kwargs=chain_type_kwargs
        # )

        prompt_stuff = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        retriever_chain = create_history_aware_retriever(self.llm, self.retriever, self.prompt)
        stuff_documents_chain = create_stuff_documents_chain(self.llm, prompt_stuff)
        conversation_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

        return conversation_rag_chain

    def retrieval_qa_inference(self, question, verbose=True):
        #query = {"query": question}
        #answer = self.retrieval_qa_chain(query)

        conversation_rag_chain = self.retrieval_qa_chain
        answer = conversation_rag_chain.invoke({
            "chat_history": self.chatHistory,
            "input": question
        })

        sources = self.list_top_k_sources(answer, k=2)

        if verbose:
            print(sources)

        return answer['answer'], sources

    def list_top_k_sources(self, answer, k=2):
        sources = [
            f'[{res.metadata["title"]}]({res.metadata["url"]})'
            for res in answer["context"]
        ]

        distinct_sources = []
        distinct_sources_str = ""

        if sources:
            k = min(k, len(sources))
            distinct_sources = list(zip(*collections.Counter(sources).most_common()))[0][:k]
            distinct_sources_str = "  \n- ".join(distinct_sources)

        if len(distinct_sources) == 1:
            return f"Sources found :  \n- {distinct_sources_str}"

        elif len(distinct_sources) > 1:
            return f"Sources {len(distinct_sources)}  :  \n- {distinct_sources_str}"

        else:
            return "No resources found to answer your question"

    def get_conversation_summary(self, historyMessages):
        # print(self.chatHistory)
        history = ChatMessageHistory()
        history.messages = historyMessages
        memory = ConversationSummaryMemory.from_messages(
            llm=self.llm,
            chat_memory=history,
            return_messages=False
        )
        return memory.buffer

