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
        self.model_name = "mxbai-embed-large"
        self.model = OllamaEmbeddings(model=self.model_name)
        if os.path.exists(path):
            self.faiss_index = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)
        else:
            self.faiss_index = None

    def calculate_ranges(self, chunks, num):
        ranges = []
        for i in range(0, len(chunks), num): 
            ranges.append(chunks[i:i + num])
        return ranges 
    
    def ranges_per_lambda(self, chunks, num_lambdas=num):
        ranges = []
        chunk_size = len(chunks) // num_lambdas
        if chunk_size == 0:
            chunk_size += 1
        for i in range(len(chunks)):
            ranges.append(chunks[i:i + chunk_size])

        return ranges

    def load_previous_indexes(self):
        # In case that we have the previous indexes we load them into the vdb
        if os.path.exists(path):
            logging.info("Loading old indexes")
            self.vectordb = FAISS.load_local(path, self.model, allow_dangerous_deserialization=True)

    def store_indexes(self, texts):

        def map_generating_embeddings(texts, model_name):
            from langchain_ollama import OllamaEmbeddings
            
            result = []
            for t in texts:
                if isinstance(t, list):
                    result.extend(t)
                else:
                    result.append(t)
                    
            logging.info(f"Loading embedding model: {model_name}")
            model = OllamaEmbeddings(model=model_name)
            logging.info("Embedding model generated")
            
            logging.info("Generating Embeddings")
            return zip(result, model.embed_documents(result))
    
        def reduce_generating_embeddings(text_embeddings):
            logging.info("All embeddings generated")
            return [item for sublist in text_embeddings for item in sublist]


        fexec = FunctionExecutor(runtime_memory=1024)
        fexec.map_reduce(map_generating_embeddings, (self.ranges_per_lambda(texts), self.model_name), reduce_generating_embeddings, spawn_reducer=0)
        results = fexec.get_result()
        
        if self.faiss_index:
            logging.info(f"{path} already created")
            self.faiss_index.add_embeddings(results)
        else:
            logging.info(f"Creating {path}")
            self.faiss_index = FAISS.from_embeddings(
                results,
                self.model
            ) 

        logging.info(f"Saving indexes on {path}")
        self.faiss_index.save_local(path)
    
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
                            content = await document.read()
                            doc = fitz.open(stream=content, filetype=document.content_type)
                            full_text = ""
                            for page in doc:
                                full_text += page.get_text()

                        case "txt":
                            full_text = await document.read()
                            full_text = full_text.decode("utf-8")

                        case "xlsx" | "csv":
                            if doc_type == "xlsx":
                                df = await pd.read_excel(f"{os.getcwd()}/data/{document.filename}")

                            else:
                                df = await pd.read_csv(f"{os.getcwd()}/data/{document.filename}")

                            result = df.to_json(orient="table", index=False)
                            parsed = loads(result)
                            del parsed["schema"]
                            full_text = dumps(parsed, indent=4)

                        # TODO aks if necessairy because an OCR method is needed and the lambdas are not going to support the .yml
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
                            treated_chunks.append(chunk.page_content)
                        return treated_chunks

                    def reduce_treated_documents(documents):
                        return [doc for chunk_list in documents for doc in chunk_list]

                    try:
                        chunks_per_lambda = vectordb.calculate_ranges(chunks, 8)
                        fexec = FunctionExecutor()
                        if len(chunks_per_lambda) >= num:
                            results = []
                            for i in range(0, len(chunks_per_lambda), num):
                                fexec.map_reduce(obtain_treated_documents, chunks_per_lambda[i:i + num], reduce_treated_documents, spawn_reducer=0)
                                results.extend(fexec.get_result())
                        else:
                            fexec.map_reduce(obtain_treated_documents, chunks_per_lambda, reduce_treated_documents, spawn_reducer=0)  
                            results = fexec.get_result()

                    except Exception as e:
                        logging.error(f"Catched excpetion, {e}, while treating the partitions of the document")
                        raise HTTPException(status_code=500, detail=f"Catched excpetion, {e}, while treating the partitions of the document")
                
                    try:
                        vectordb.store_indexes(results)
                        logging.info("All chunkes had been stored correctly")
                        # vectordb.load_previous_indexes()
                        # logging.info("All index stored in local")

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