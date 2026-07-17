from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatMistralAI(model="mistral-small-latest", temperature=0.9)

print("Choose your AI mode")
print("press 1. Chatbot")
print("press 2. Funny Chatbot")
print("press 3. Angry Chatbot")
print("press 4. Sad Chatbot")
choice = input("Enter your choice (1-4): ")

# Initialize messages with the selected system prompt
if choice == "1":
    messages = [SystemMessage(content="You are a helpful AI agent.")]
elif choice == "2":
    messages = [SystemMessage(content="You are a funny AI agent.")]
elif choice == "3":
    messages = [SystemMessage(content="You are an angry AI agent.")]
elif choice == "4":
    messages = [SystemMessage(content="You are a sad AI agent.")]
else:
    print("Invalid choice. Defaulting to a helpful AI agent.")
    messages = [SystemMessage(content="You are a helpful AI agent.")]

# --- FIXED: Removed the line that overwrote the 'messages' list ---

print("--------------Type 0 to quit the chat.-------------")
while True:
    user_prompt = input("YOU : ")

    # Check for exit condition immediately before appending to history
    if user_prompt == "0":
        break

    messages.append(HumanMessage(content=user_prompt))

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print(f"Bot: {response.content}")

print("\n--- Chat History ---")
print(messages)
