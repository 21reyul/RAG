import requests
import urllib.parse

information = "Estimados lectores"
encoded_info = urllib.parse.quote(information)  # codifica espacios y caracteres especiales

url = f"http://localhost:8000/store_info/{encoded_info}/"

response = requests.post(url)

print(f"Status code: {response.status_code}")
print(response.json())