from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv(override=True)


my_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Ek strong fully-supported chat model select karein
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=my_token
)

model = ChatHuggingFace(llm=llm)
response = model.invoke("write a poem on AI")
print(response.content)
