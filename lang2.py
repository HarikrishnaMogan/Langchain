from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.tools import tool

model = init_chat_model("groq:llama-3.3-70b-versatile")


# res = model.invoke("Hi, how are you?")
# print(res.content)


# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to French."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]

# conversation = [
#     {"role": "system", "content": "You are a helpful assistant that translates English to French."},
#     {"role": "user", "content": "Translate: I love programming."},
#     {"role": "assistant", "content": "J'adore la programmation."},
#     {"role": "user", "content": "Translate: I love building applications."}
# ]

# res = model.invoke(conversation)
# print(res)

# streaming
# for chunk in model.stream("why parret have colorfull feathers"):
#     print(chunk.text, end='',flush=True)

# Tool calling..........................
@tool
def get_weather(city):
    "It used to get the city weather details"
    return f"It is sunny in {city}, 30 deg"

conversation = [ 
    {"role": "system", "content": "You are a helpful assistant that tell weather details in a city using 'get_weather' tool."},
    {"role": "user", "content": "what is weather in SF?"}
]

# tool_choice force the model to execute the tools
model_with_tools = model.bind_tools([get_weather],tool_choice='any')

res = model_with_tools.invoke(conversation)
conversation.append(res)

for tool_call in res.tool_calls:
    if tool_call['name'] == 'get_weather':
        tool_result = get_weather.invoke(tool_call)
        print(tool_result) 
        conversation.append(tool_result)

        final_res = model.invoke(conversation)
        print(final_res)

