import requests
import os

url = f"http://localhost:8000/store_pdf/"

file_path = f"{os.getcwd()}/data/quijote_reducido.pdf"
print(os.path.exists(file_path))

with open(file_path, "rb") as f:
    files = {"document": (file_path, f, "application/pdf")}
    response = requests.post(url, files=files)

print(f"Status code: {response.status_code}")
