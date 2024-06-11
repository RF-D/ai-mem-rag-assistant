from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools  import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools.tavily_search import tavily_search
from langchain_anthropic import ChatAnthropic

tools = [tavily_search]
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.8)

prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. When using the search tool please include and cite your exact sources ('URL') in the response at the end"),
                                            ("human", "{input}"),
                                            ("placeholder", "{agent_scratchpad}")])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

user_input = "What is the weather in tx right now?"
response = agent_executor.invoke({"input": user_input},config={"callbacks": [tracer]})

print(response)
