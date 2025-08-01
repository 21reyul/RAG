import asyncio
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import UserMessage, ModelInfo
from autogen_agentchat.ui import Console

termination_condition = MaxMessageTermination(3)


async def main() -> None:
    model_client = OllamaChatCompletionClient(
        model="hf.co/stepii/salamandra-7b-instruct-tools-GGUF:Q8_0", 
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=True),
        #model_client_stream=True,
    )

    # define the Docker CLI Code Executor
    code_executor = DockerCommandLineCodeExecutor(work_dir="coding")

    # start the execution container
    await code_executor.start()

    code_executor_agent = CodeExecutorAgent("code_executor_agent", code_executor=code_executor)
    coder_agent = AssistantAgent("coder_agent", model_client=model_client)

    groupchat = RoundRobinGroupChat(
        participants=[coder_agent, code_executor_agent], termination_condition=termination_condition
    )

    task = "Write a Hello World in python"
    await Console(groupchat.run_stream(task=task))

    # stop the execution container
    await code_executor.stop()

asyncio.run(main())