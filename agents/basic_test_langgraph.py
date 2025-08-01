from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

llm = init_chat_model(
    #"ollama:hf.co/stepii/salamandra-7b-instruct-tools-GGUF:Q8_0"
    "ollama:llama3.1:8b"
)

def get_language_prompt(language: str) -> str:
    return f"""
        You are a helpful and expert programming assistant specialized in {language}.
        Your job is to answer any questions related to {language} clearly, accurately, and concisely.
        If the user's query is ambiguous or lacks context, ask clarifying questions before answering.
        Do not respond to questions unrelated to {language}.
    """.strip()



class MessageClassifier(BaseModel):
    message_type: Literal["PythonExpert", "JavaExpert", "CExpert", "VersatileCoder"] = Field(
        ...,
        description="Classify if the message requires a response from a PythonEXpert a JavaExpert a CExpert or a VersatileCoder"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role" : "system", 
            "content" : """Classify the user message as either:
                -PythonExpert: if it ask a question related to python
                -JavaExpert: if it ask a question related to java
                -CExpert: if it ask a question related to c
                -VersatileCoder: if it ask a question related to coding but the language indicated is not related to python, c or java
            """
        }, 
        {
            "role": "user", 
            "content": last_message.content
        }
    ])

    return {"message_type": result.message_type}


#TODO establish the next argument 
def router(state: State): 
    message_type = state.get("message_type", "VersatileCoder")
    if message_type == "PythonExpert":
        return {"next": ""}
    
    elif message_type == "JavaExpert":
        return {"next": ""}
    
    elif message_type == "CExpert":
        return {"next": ""}
    
    return {"next": ""}

def python_agent(state: State):
    last_message = state["messages"][-1]
    reply = llm.invoke([
        {
            "role": "system", 
            "content": get_language_prompt("python")
        }, 
        {
            "role": "user", 
            "content": last_message.content
        }
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def java_agent(state: State):
    last_message = state["messages"][-1]
    reply = llm.invoke([
        {
            "role": "system", 
            "content": get_language_prompt("java")
        }, 
        {
            "role": "user", 
            "content": last_message.content
        }
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def c_agent(state: State):
    last_message = state["messages"][-1]
    reply = llm.invoke([
        {
            "role": "system", 
            "content": get_language_prompt("C")
        }, 
        {
            "role": "user", 
            "content": last_message.content
        }
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def versatile_agent(state: State):
    last_message = state["messages"][-1]
    reply = llm.invoke([
        {
            "role": "system", 
            "content": """
                Generate the response based on the language indicate on the content that the user indicates you.
            """
        }, 
        {
            "role": "user", 
            "content": last_message.content
        }
    ])
    return {"messages": [{"role": "assistant", "content": reply.content}]}

graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("python", python_agent)
graph_builder.add_node("java", java_agent)
graph_builder.add_node("c", c_agent)
graph_builder.add_node("versatile", versatile_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_edge("router", "python")
graph_builder.add_edge("router", "java")
graph_builder.add_edge("router", "c")
graph_builder.add_edge("router", "versatile")

graph_builder.add_edge("python", END)
graph_builder.add_edge("java", END)
graph_builder.add_edge("c", END)
graph_builder.add_edge("versatile", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()