# 1 Installation
We have to create a python enviornment:
    
    python -m venv .venv
Also you have to execute create_storage.sh, this will create you a bucket into your /tmp/lithops$USER directory, use the following command:

    ./create_storage.sh
# 2 Usage
We have to enter into the enviornment created and execute the following command:

    pip install \
    transformers \
    torch \
    langchain \
    langchain-community \
    langchain-huggingface \
    sentence-transformers \
    lithops \
    fastapi \
    uvicorn \
    numpy \
    requests \
    faiss-cpu \
    pymupdf \
    ollama \
    python-multipart \
    pip install "autogen-ext[ollama]" \
    --upgrade
# 3 Execution
In one hand you have to execute the store_data file with the following command, to enable FastAPI petitions:

    uvicorn store_data:app <--reload>

And in the other hand you can create your own client to make some petitions to it, as an example you have client.py

In case you want to execute any querying method you have to do it this way:

    python3 -m querying.<script>
