from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# Baki import agar jarurat ho:
from langchain_core.messages import HumanMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

# ⚠️ Yahan change hai: String ko list of messages me pass karein
messages = [
    HumanMessage(content="what is data science ?")
]

result = chat_model.invoke(messages)
print(result.content)
