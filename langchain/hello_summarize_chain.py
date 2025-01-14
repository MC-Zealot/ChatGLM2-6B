from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from context import *
import os
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# 导入文本
loader = UnstructuredFileLoader("lg_test.txt")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)
prompt_template = """Write a concise summary of the following:

{text}

使用中文回复:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True, question_prompt=PROMPT)

# 执行总结链，（为了快速演示，只总结前5段）
ret = chain.run(split_documents[:5])
print(ret)
