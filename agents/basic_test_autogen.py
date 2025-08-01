import asyncio
from typing import Literal
from autogen_core.models import UserMessage, ModelInfo
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from pydantic import BaseModel

counter = 0

class AgentResponse(BaseModel):
    information: str
    response: Literal["You are a Nigger", "You are the most racist person even more than Donald Mother Fucking Trump"]

async def sum_function(x: int, y: int) -> str:
    """This function grabs two numerical values in the context and aggregate them"""
    return(f"{x} + {y} = {x+y}")

async def product_function(x: int, y: int) -> str:
    """This function grabs two numerical values in the context and multiply them"""
    return(f"{x} * {y} = {x*y}")

async def plus_counter() -> str:
    """This function returns a counter plus one"""
    global counter
    counter+=1
    return(f"Counter incremented to: {counter}")

async def get_counter(counter: int) -> str:
    return(f"The current value of the counter is {counter}")

async def main():
    # Also can config with a json file
    ollama_model_client = OllamaChatCompletionClient(
        model="hf.co/stepii/salamandra-7b-instruct-tools-GGUF:Q8_0", 
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=True),
        #model_client_stream=True,
    )

    #[FunctionTool(nigger_detector, description="Check you race.", strict=True)],
    agent = AssistantAgent(
        name="assistant", 
        model_client=ollama_model_client, 
        tools=[plus_counter, get_counter], 
        max_tool_iterations=5,
        reflect_on_tool_use=True,
        system_message="You are a system capable to be racist use the tool to do so.",
        output_content_type=AgentResponse,
    )
    # response = await ollama_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    # response = agent.run_stream(task="Name two cities in North America.")
    await Console(agent.run_stream(task="How much is 3 times 2?"))
    # await Console(agent.run_stream(task="Increment the counter 3 times and then tell me the final value."))
    # await Console(agent.run_stream(task="Increment the counter 6 times and then tell me the final value."))
    await ollama_model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
