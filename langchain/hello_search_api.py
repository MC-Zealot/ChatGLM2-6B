search_api="8281071f58b19ce8040b4c648165485cdab14b75dbf4c2eee86f40e1ae561428"
openai_key=""
import os
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"] = search_api

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

# 加载 OpenAI 模型
llm = OpenAI(temperature=0,max_tokens=2048)

 # 加载 serpapi 工具
tools = load_tools(["serpapi"])

# 如果搜索完想再计算一下可以这么写
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
# tools=load_tools(["serpapi","python_repl"])

# 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行 agent
#ret = agent.run("What's the date today? What great events have taken place today in history?")
ret = agent.run("What's the date today?What important events have taken place today in history?")
#print(ret)
