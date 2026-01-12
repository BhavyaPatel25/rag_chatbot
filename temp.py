from retriever import retrieve_context

question = "What is Bhavya Patel's experience?"
context = retrieve_context(question)

print("Retrieved context:")
print("-" * 40)
print(context)
