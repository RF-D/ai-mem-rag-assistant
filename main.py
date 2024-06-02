
import os
from dotenv import load_dotenv
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
from langchain_community.document_loaders import FireCrawlLoader

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
v_api_key = os.getenv("VOYAGE_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# VoyageAI Setup
embeddings = VoyageAIEmbeddings(
    voyage_api_key=v_api_key,
    model="voyage-large-2-instruct"
)

# FireCrawl Setup
loader = FireCrawlLoader(
    api_key=firecrawl_api_key, url="https://python.langchain.com/v0.1/docs/get_started/introduction/", mode="scrape"
)

data = loader.load()
print(data)

# Chat setup
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
