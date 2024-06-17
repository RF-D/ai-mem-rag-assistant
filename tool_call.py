from utils.env_loader import load_env_vars
#from tools.tavily_search import tavily_search
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tools.tavily_search_results import tavily_search_r
from tools.tavily_search import tavily_search
from tools.scrape_search import  scrape_search







tavily_api_key = load_env_vars()[5]
anthropic_api_key = load_env_vars()[0]

tools = [tavily_search]
model = ChatAnthropic(model="claude-3-sonnet-20240229")

agent_executor = create_react_agent(model, tools)


response = agent_executor.invoke({"messages":[HumanMessage(content="What is the weather in TX?")]})

response["messages"]
 
print(response["messages"])

# # print(tools)

# prompt = ChatPromptTemplate.from_template(
#     """Answer the question based only on the context provided.

# Context: {context}

# Question: {question}"""
# )
# chain = (
#     RunnablePassthrough.assign(context=(lambda x: x["question"]) | tools )
#     | prompt
#     | model
#     | StrOutputParser()
# )

# response = chain.invoke({"question": "What happened to roaring kitty this weekend?"})
# print(response)


