from langchain_pinecone import PineconeVectorStore
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic
from utils.env_loader import load_env_vars
from tools.firecrawl_scrape_loader import scrape
from tools.firecrawl_crawl_loader import crawl
from tools.text_splitter import split_text
from tools.voyage_embeddings import setup_voyageai
from tools.retriever_tool import retriever_tool


# Load environment variables
anthropic_api_key, v_api_key, firecrawl_api_key, pinecone_api_key = load_env_vars()

# VoyageAI Setup
embeddings = setup_voyageai("voyage-large-2-instruct")

# Use Firecrawl to scrape or crawl URL
data = scrape("https://www.youtube.com/watch?v=GXRveOki4kE")

# Split text into documents
docs = split_text(data)


vectorstore = PineconeVectorStore.from_documents(
    docs, embeddings, index_name="claude01")

retriever = retriever_tool(vectorstore)

retrieved_docs = retriever.invoke(
    "What is the youtube url of this video")

print(retrieved_docs[0].page_content)
# Chat setup
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.8)

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
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

print("Welcome to the ChatBot powered by Anthropic's Claude! Type 'exit' to end the conversation.")

# Base Termnial Chat Loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Thank you for chatting. Goodbye!")
        break

    ai_response = conversation.predict(input=user_input)
    print(f"Claude: {ai_response}")
