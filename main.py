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
from dotenv import load_dotenv

load_dotenv()

os.getenv("ANTHROPIC_API_KEY")

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

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thank you for chatting. Goodbye!")
        break

    ai_response = conversation.predict(input=user_input)
    print(f"Claude: {ai_response}")
