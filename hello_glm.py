import os
import langchain as lc

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).quantize(8).cuda()
chatglm=model.eval()

response, history = chatglm.chat(tokenizer, "怎么评价人工智能")
print(response)
