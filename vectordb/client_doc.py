# import requests
# import os

# url = f"http://localhost:8000/upload_documents/"

# file_path = [f"{os.getcwd()}/data/vectords.pdf"]

# for path in file_path:
#     with open(path, "rb") as f:
#         files = {"document": (path, f, "application/pdf")}
#         response = requests.post(url, files=files)

# print(f"Status code: {response.status_code}")

import requests
import os

url = "http://localhost:8000/upload_documents/"

file_paths = [f"{os.getcwd()}/data/vectords.pdf"]

files = []
for path in file_paths:
    f = open(path, "rb")
    files.append(("documents", (os.path.basename(path), f, "pdf")))

response = requests.post(url, files=files)

print(f"Status code: {response.status_code}")
print(response.text)
