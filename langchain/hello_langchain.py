from context import *
import os
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
#from langchain.llms import OpenAI
import langchain as lc

llm = lc.llms.OpenAI(model_name="text-davinci-003",max_tokens=1024)
ret = llm("怎么评价人工智能")
print(ret)
