FROM python:3.12

RUN pip install --upgrade pip

RUN pip install langchain \
    langchain-community \
    langchain-huggingface \
    langchain-core \
    langchain-community pypdf \
    pika \
    faiss-cpu \
    sentence-transformers \
    sentencepiece \
    protobuf

ENV APP_HOME=/home/query
RUN useradd -m -d /home/query query
WORKDIR $APP_HOME

COPY store_data.py $APP_HOME
RUN chown query:query $APP_HOME/store_data.py 
USER query

CMD python3 "$APP_HOME/store_data.py"

# TODO make volumes to persist faiss_index in case it exists in the docker instance