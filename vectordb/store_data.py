from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from lithops import FunctionExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from json import loads, dumps
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

# Accepted documents on the upload file phase
accepted_documents = ["txt", "pdf", "xlsx", "csv", "jpg", "png"]


# TODO change the storage with related_subject to current subject when you have the agent per subject implementation ready
class DataStorage():
    def __init__(self):
        # TODO pull in case its not in local
        self.model = OllamaEmbeddings(model="mxbai-embed-large")
        if os.path.exists(path):
            self.faiss_index = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)
        else:
            self.faiss_index = None

    def load_previous_indexes(self):
        # In case that we have the previous indexes we load them into the vdb
        if os.path.exists(path):
            logging.info("Loading old indexes")
            self.vectordb = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)

    def store_indexes(self, documents):
        logging.info("Storing indexes")
        if self.faiss_index:
            self.faiss_index.add_documents(documents)
        else:
            self.faiss_index = FAISS.from_documents(documents, self.model)    
        
        self.faiss_index.save_local(path)

    def calculate_ranges(self, chunks, num):
        ranges = []
        for i in range(0, len(chunks), num): 
            ranges.append(chunks[i:i + num])
        return ranges 
    
    async def reading_pdf(self, document = File(...)):
        content = await document.read()
        doc = fitz.open(stream=content, filetype=document.content_type)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    @app.post("/upload_documents/")
    async def upload_documents(documents: list[UploadFile] = File(...)):
        # TODO treat multiple documents in different process in the local machine
        for document in documents:
            doc_type = document.content_type
            if doc_type not in accepted_documents:
                raise HTTPException(status_code=400, detail=f"You are not allowed to upload this files: {doc_type}")
            
            else:
                try:
                    vectordb = DataStorage()
                    logging.info(f"Treating {document.filename}")
                    match doc_type:
                        case "pdf":
                            full_text = await vectordb.reading_pdf(document)

                        case "txt":
                            full_text = await document.read()
                            full_text = full_text.decode("utf-8")

                        case "xlsx" | "csv":
                            if doc_type = "xlsx":
                                df = await pd.read_excel(document)
                            else:
                                df = pd.read_csv(f"{os.getcwd()}/data/{document.filename}")

                            result = df.to_json(orient="table")
                            parsed = loads(result)
                            del parsed["schema"]
                            full_text = dumps(parsed, indent=4)

                        case "jpg" | "png":
                            pass

                    try:
                        chunker = RecursiveCharacterTextSplitter(
                            chunk_size=800,  
                            chunk_overlap=0,  
                            separators=["\n\n", "\n", ".", "?", "!", " ", ""] 
                        )
                        chunks = chunker.create_documents([full_text])
                        logging.info(f"{document.filename} has been chunked")

                    except Exception as e:
                        logging.error(f"It ocurred the following error, {e}, while trying to chunk the document")
                        raise HTTPException(status_code=500, detail="There was an error while trying to chunk the document")

                    def obtain_treated_documents(chunks):
                        treated_chunks = []
                        for chunk in chunks:
                            treated_chunks.append(Document(page_content=chunk.page_content))
                        return treated_chunks

                    def reduce_treated_documents(documents):
                        return [doc for chunk_list in documents for doc in chunk_list]

                    try:
                        chunks_per_lambda = vectordb.calculate_ranges(chunks, 8)
                        fexec = FunctionExecutor()
                        if len(chunks_per_lambda) >= num:
                            results = []
                            for i in range(0, len(chunks_per_lambda), num):
                                fexec.map_reduce(obtain_treated_documents, chunks_per_lambda[i:i + num], reduce_treated_documents)
                                results.extend(fexec.get_result())
                        else:
                            fexec.map_reduce(obtain_treated_documents, chunks_per_lambda, reduce_treated_documents)  
                            results = fexec.get_result()

                    except Exception as e:
                        logging.error(f"Catched excpetion, {e}, while treating the partitions of the document")
                        raise HTTPException(status_code=500, detail="Catched excpetion, {e}, while treating the partitions of the document")
                
                    try:
                        # TODO optimize the index storage
                        #vectordb.store_indexes(results)
                        logging.info("All chunkes had been stored correctly")
                        vectordb.load_previous_indexes()
                        logging.info("All index stored in local")

                    except Exception as e:
                        logging.error(rf"Catched exception: {e}")
                        raise HTTPException(status_code=500, detail="Error storing the document")
                    
                except Exception as e:
                    logging.error(f"Catched exception: {e}")
                    raise HTTPException(status_code=500)

    
    @app.post("/store_info/")
    def store_documents(information: str):
        docs = []
        vectordb = DataStorage()
        docs.append((Document(page_content=information)))
        vectordb.store_indexes(docs)    