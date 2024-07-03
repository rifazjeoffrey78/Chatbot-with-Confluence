import sys
import logging
import shutil

sys.path.append('../')
from config import (CONFLUENCE_SPACE_NAME, CONFLUENCE_SPACE_KEY, CONFLUENCE_BASE_URL,
                    CONFLUENCE_USERNAME, CONFLUENCE_PASSWORD, CONFLUENCE_API_KEY, PERSIST_DIRECTORY)

from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from llama_index.readers.confluence import ConfluenceReader
from langchain_community.vectorstores import Chroma


class DataLoader():
    """Create, load, save the DB using the confluence Loader"""
    def __init__(
        self,
        confluence_url=CONFLUENCE_SPACE_NAME,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_PASSWORD,
        api_key=CONFLUENCE_API_KEY,
        space_key=CONFLUENCE_SPACE_KEY,
        persist_directory=PERSIST_DIRECTORY
    ):

        self.confluence_url = confluence_url
        self.username = username
        self.password = password
        self.api_key = api_key
        self.space_key = space_key
        self.persist_directory = persist_directory

    def load_from_confluence_loader(self):
        """Load HTML files from Confluence"""
        # loader = ConfluenceLoader(
        #     url=self.confluence_url,
        #     username=self.username,
        #     api_key=self.password,
        #     cloud=False,
        #     space_key=self.space_key,
        #     limit=50,
        # )
        # documents = loader.load()

        reader = ConfluenceReader(base_url=CONFLUENCE_BASE_URL)
        documents = reader.load_data(
            space_key=self.space_key,
            # page_ids=["7733953"],
            include_attachments=False,
            page_status="current",
            # start=0,
            # max_num_results=5,
        )
        return documents

    def split_docs(self, docs):
        # Markdown
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Split based on markdown and add original metadata
        md_docs = []
        for doc in docs:
            md_doc = markdown_splitter.split_text(doc.text)#doc.page_content
            for i in range(len(md_doc)):
                md_doc[i].metadata = md_doc[i].metadata | doc.metadata
            md_docs.extend(md_doc)
            #print(md_doc)

        # RecursiveTextSplitter
        # Chunk size big enough
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )

        splitted_docs = splitter.split_documents(md_docs)
        return splitted_docs

    def save_to_db(self, splitted_docs, embeddings):
        """Save chunks to Chroma DB"""
        db = Chroma.from_documents(splitted_docs, embeddings, persist_directory=self.persist_directory)
        db.persist()
        return db

    def load_from_db(self, embeddings):
        """Loader chunks to Chroma DB"""
        db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        return db

    def set_db(self, embeddings):
        """Create, save, and load db"""
        try:
            print("Removing Chroma DB directory")
            #shutil.rmtree(self.persist_directory)
        except Exception as e:
            logging.warning("%s", e)

        # Load docs
        docs = self.load_from_confluence_loader()

        # Split Docs
        splitted_docs = self.split_docs(docs)

        # Save to DB
        db = self.save_to_db(splitted_docs, embeddings)

        return db

    def get_db(self, embeddings):
        """Create, save, and load db"""
        db = self.load_from_db(embeddings)
        return db


if __name__ == "__main__":
    x = DataLoader()
    docs = x.load_from_confluence_loader()
    x.split_docs(docs)
    #pass
