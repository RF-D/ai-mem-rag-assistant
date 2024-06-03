from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
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


# Load environment variables
anthropic_api_key, v_api_key, firecrawl_api_key, pinecone_api_key = load_env_vars()


data = scrape("www.google.com")


docs = split_text(data)

# VoyageAI Setup
embeddings = VoyageAIEmbeddings(
    voyage_api_key=v_api_key,
    model="voyage-large-2-instruct"
)

index_name = "claude01"
docsearch = PineconeVectorStore.from_documents(
    docs, embeddings, index_name=index_name)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke(
    "What open-source libraries does the framework consist of?")

print(retrieved_docs[0].page_content)
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
