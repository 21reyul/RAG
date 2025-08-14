import os
import requests
import logging
import urllib.parse

class Client():
    def __init__(self):
        self.url = "http://localhost:8000/"
        
    def post_doc(self, documents, method="upload_document"):
        
        url = self.url + method
        
        file_paths = [f"{os.getcwd()}/data/{document}" for document in documents]
        extensions = [document.split(".")[1] for document in documents]

        files = []
        for path, extension in zip(file_paths, extensions):
            if os.path.exists(path):
                logging.info(f"Treating document: {path.split("/")[-1]}")
                print(f"Treating document: {path.split("/")[-1]}")
                f = open(path, "rb")
                files.append(("documents", (os.path.basename(path), f, extension)))
            else:
                logging.info(f"The following file doesn't exists: {path.split("/")[-1]}")
                print((f"The following file doesn't exists: {path.split("/")[-1]}"))

        requests.post(url, files=files)

    def post_info(self, info, method="upload_info"):
        url = self.url + method + f"/{urllib.parse.quote(info)}"
        print(url)
        requests.post(url)


if __name__ == "__main__":
    client = Client()
    #client.post_doc(["reduced_pride_and_prejudice.pdf", "barreim.pdf"])
    client.post_info("Pride and Prejudice was written by Jane Austen")