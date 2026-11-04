from autonomousds.llm.models import get_chat_model

model = get_chat_model(agent="smoke_test")
response = model.invoke("In one sentence, what is cross-validation?")
print(response.text)
print(response.usage_metadata)