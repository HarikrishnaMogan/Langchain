# pip install -qU "langchain[anthropic]" to call the model
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,ToolMessage
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field
import os
load_dotenv()

class WeatherInput(BaseModel):
    city:str =Field(description='The city to look for weather')


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # print(runtime)
    return f"The current weather in {city} is 30 degree, sunny"

llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""You are a helpful weather assistant.If the user query is incomplete, ask follow-up questions to collect missing information.
    Tools you have access to:
    get_weather
    Dont mention anything about tool to the user. Just ask questions and answer
"""
)

message = {"messages": [{"role": "user", "content": "what is the weather "}]}


# Run the agent
res = agent.invoke(message)
# print(res)

messages = res['messages']  # or res.messages if it's an object

# Extract AI messages
ai_responses = [msg.content for msg in messages if isinstance(msg, AIMessage)]

# Extract Tool messages
tool_responses = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

# Join if multiple messages
final_ai_response = "\n".join(ai_responses)
final_tool_response = "\n".join(tool_responses)
message['messages'].append({"role":"ai","content":final_ai_response})
print("Tool Output:\n", final_tool_response)
print("\nAI Output:\n", final_ai_response)


message["messages"].append({"role": "user", "content": "city is sf"})


# Run the agent
res = agent.invoke(message)
# print(res)

messages = res['messages']  # or res.messages if it's an object

# Extract AI messages
ai_responses = [msg.content for msg in messages if isinstance(msg, AIMessage)]
# Extract Tool messages
tool_responses = [msg.content for msg in messages if isinstance(msg, ToolMessage)]

# Join if multiple messages
final_ai_response = "\n".join(ai_responses)
final_tool_response = "\n".join(tool_responses)

print("Tool Output:\n", final_tool_response)
print("\nAI Output:\n", final_ai_response)