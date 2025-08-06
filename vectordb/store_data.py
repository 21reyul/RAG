from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from lithops import FunctionExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
import os
import fitz

app = FastAPI()
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

path = "/tmp/.faiss_index"

# This number is the maximum lambdas that you want to use in order to store the indexs of the documents, you can change it on your own
num = 10

# TODO change the storage with related_subject to current subject when you have the agent per subject implementation ready
class DataStorage():
    def __init__(self):
        # TODO pull in case its not in local
        self.model = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        self.load_previous_indexes()

    def load_previous_indexes(self):
        # In case that we have the previous indexes we load them into the vdb
        if os.path.exists(path):
            logging.info("Loading old indexes")
            self.vectordb = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)

    def store_indexes(self, documents):
        if os.path.exists(path):
            faiss_index = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)
            faiss_index.add_documents(documents)
        else:
            faiss_index = FAISS.from_documents(documents, self.model)    
        
        faiss_index.save_local(path)

    def calculate_ranges(self, chunks, num):
        ranges = []
        for i in range(0, len(chunks), num): 
            ranges.append(chunks[i:i + num])
        return ranges 

    # TODO add multimodal files and HTTP exceptions
    @app.post("/store_pdf/")
    async def store_pdf_bucket(document: UploadFile = File(...)):
        if document.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="You are only allowed to upload pdf files")
        try:
            content = await document.read()
            logging.info(f"Treating {document.filename}")
            vectordb = DataStorage()
            doc = fitz.open(stream=content, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            d = Document(page_content=full_text)

            chunker = RecursiveCharacterTextSplitter(
                chunk_size=800,  
                chunk_overlap=0,  
                separators=["\n\n", "\n", ".", "?", "!", " ", ""] 
            )
            chunks = chunker.split_documents([d])
            logging.info(f"{document.filename} has been chunked")

            def obtain_treated_documents(chunks):
                treated_chunks = []
                for chunk in chunks:
                    treated_chunks.append(Document(page_content=chunk.page_content))
                return treated_chunks

            def reduce_treated_documents(documents):
                return [doc for chunk_list in documents for doc in chunk_list]

            chunks_per_lambda = vectordb.calculate_ranges(chunks, 8)
            fexec = FunctionExecutor()
            if len(chunks_per_lambda) >= num:
                results = []
                for i in range(0, len(chunks_per_lambda), num):
                    futures = fexec.map_reduce(obtain_treated_documents, chunks_per_lambda[i:i + num], reduce_treated_documents)
                    fexec.wait(futures)  
                    results = fexec.get_result(futures)
                    vectordb.store_indexes(results)
            else:
                futures = fexec.map_reduce(obtain_treated_documents, chunks_per_lambda, reduce_treated_documents)
                fexec.wait(futures)  
                results = fexec.get_result(futures)

            vectordb.store_indexes(results)
            logging.info("All chunkes had been stored correctly")
            vectordb.load_previous_indexes()
            logging.info("All index stored in local")

        except Exception as e:
            logging.error(rf"Catched exception: {e}")

        except Exception as e:
            logging.error(rf"Catched exception: {e}")
            raise HTTPException(status_code=500, detail="Error storing the document")
    
    @app.post("/store_info/{information}")
    def store_documents(information: str):
        docs = []
        vectordb = DataStorage()
        docs.append((Document(page_content=information)))
        vectordb.store_indexes(docs)    