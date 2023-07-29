from context import *
import os
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone
embeddings = OpenAIEmbeddings()
# 初始化 pinecone
pinecone.init(
  api_key=PIPECONE_KEY,
  environment=PIPECONE_ENV
)

loader = DirectoryLoader('.', glob='keda4.txt')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

index_name="zzz"

# 持久化数据
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# 加载数据
docsearch = Pinecone.from_existing_index(index_name,embeddings)

query = "科大讯飞今年第一季度收入是多少？"
query = "科大讯飞2021年营业收入是多少？科大讯飞2022年营业收入是多少？科大讯飞2021和2022年2年营业总收入是多少？2006年至2021年，公司年复合增长是多少?"
query = "科大讯飞2022年营业收入是多少？同比增长了多少?能否推导2021年的应收？推导公式：2021年总营收=2022年的总应收/(1+同比增长), 其中同比增长要换算成小数的形式"
#docs = docsearch.similarity_search(query, include_metadata=True)
docs = docsearch.similarity_search(query)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
ret = chain.run(input_documents=docs, question=query)
print(ret)
#query = "2006年至2021年，公司营年复合增长是多少？"
#docs = docsearch.similarity_search(query, include_metadata=True)

