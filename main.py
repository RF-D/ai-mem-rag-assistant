
import os
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
v_api_key = os.getenv("VOYAGE_API_KEY")

embeddings = VoyageAIEmbeddings(
    voyage_api_key=v_api_key,
    model="voyage-large-2-instruct"
)

chat = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.8)

# Set up the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant named Claude. Answer the user's questions to the best of your ability."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Initialize the memory
memory = ConversationBufferMemory(return_messages=True)

# Create the conversation chain
conversation = ConversationChain(memory=memory, prompt=prompt, llm=chat)

print("Welcome to the ChatBot powered by Anthropic's Claude! Type 'exit' to end the conversation.")

# Base Termnial Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thank you for chatting. Goodbye!")
        break

    ai_response = conversation.predict(input=user_input)
    print(f"Claude: {ai_response}")
