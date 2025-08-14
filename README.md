# Installation & Setup

## 1. Local Environment Setup

If you cloned this repository locally, you can create a Python virtual environment with:

```bash
python -m venv .venv
```

Once the environment is created, install the required dependencies:

```bash
pip install langchain \
    langchain-community \
    langchain-ollama \
    lithops \
    fastapi \
    pandas \
    PyMuPDF
```
Make sure Python is installed on your system. You can check it by running:

```bash
python --version
```

## 2. Conda Environment (for PyRun)
If you are using this RAG pipeline on PyRun, make sure your environment.yml contains the following dependencies:

    name: base
    channels:
        - conda-forge
    dependencies:
        - python=3.12
        - ipykernel
        - ipywidgets
        - git
        - faiss-cpu
        - pip
        - curl
        - pip:
            - $LITHOPS
            - langchain
            - langchain-community
            - langchain-ollama
            - lithops
            - fastapi
            - pandas
            - PyMuPDF

## 3. Ollama Installation
You also need to install Ollama on your machine.

### Local machine
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
After installing Ollama, pull the required models:
```bash
ollama pull <model>
```
> *Note: You need both a generative and an embedding model for the pipeline to work correctly. Ollama supports GPU usage, which is recommended, although embedding generation will be slower on CPU.*

### PyRun / AWS EC2
1. SSH into your EC2 instance:
```bash
ssh -i /path/to/your-key.pem ec2-user@<public-ip>
```
2. Follow the same steps as above to install Ollama and pull the models.

# Execution

1. Start the FastAPI server by running:
```bash
uvicorn store_data:app <--reload>
```
2. You can now store data in the vector database and query it.

3. For examples, check the rag_pipeline notebook, which demonstrates how to store information and execute queries on the vector database.

> *Note: You need to have some documents on data folder to store information otherwise it won't work, check it out.*

