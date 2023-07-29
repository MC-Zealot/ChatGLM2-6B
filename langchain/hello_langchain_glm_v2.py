from context import *
import os
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).quantize(8).cuda()
# chatglm=model.eval()

from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import ChatGLM
llm = ChatGLM(endpoint_url='http://127.0.0.1:8000')

import pinecone
embeddings = OpenAIEmbeddings()
# 初始化 pinecone
pinecone.init(
  api_key=PIPECONE_KEY,
  environment=PIPECONE_ENV
)

#向量库
embeddings = OpenAIEmbeddings()
#加载文件
loader = DirectoryLoader('.', glob='keda.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

index_name="zzz"

# 持久化数据
# docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)
# 加载数据
docsearch = Pinecone.from_existing_index(index_name,embeddings)

query = "科大讯飞今年第一季度收入是多少？"
query = "2022年，科大讯飞营业收入是多少？科大讯飞今年第一季度收入是多少？"
# query = "科大讯飞2021年营业收入是多少？科大讯飞2022年营业收入是多少？科大讯飞2021和2022年2年营业总收入是多少？2006年至2021年，公司年复合增长是多少?"
# query = "科大讯飞2022年营业收入是多少？同比增长了多少?能否推导2021年的应收？推导公式：2021年总营收=2022年的总应收/(1+同比增长), 其中同比增长要换算成小数的形式"
#docs = docsearch.similarity_search(query, include_metadata=True)
docs = docsearch.similarity_search(query, k=4)

# chain = load_qa_chain(chatglm, chain_type="stuff", verbose=True)
prompt= f"已知信息： \n{'n'.join([t.page_content for t in docs])}\n根据已知信息回答问题：\n{query}"
# print("prompt: ",prompt)
# response, history = chatglm.chat(tokenizer, prompt)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
ret = chain.run(input_documents=docs, question=query)
print(ret)

# curl -X POST "http://127.0.0.1:8501" -H 'Content-Type: application/json' -d '{"prompt": "你好", "history": []}'