import os
import streamlit as st
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from utils.llm_manager import LLMManager
from tools.retriever_tools import retriever_tool_meta
from langchain_pinecone import PineconeVectorStore
from tools.voyage_embeddings import vo_embed
from dotenv import load_dotenv
from streamlit_ace import st_ace

# Load environment variables and set up LLM
load_dotenv()

LLMManager.initialize_ollama_models()

# Set up vector store and retriever
@st.cache_resource
def setup_vectorstore():
    return PineconeVectorStore.from_existing_index(embedding=vo_embed(), index_name="langchain")

vectorstore = setup_vectorstore()
retriever = retriever_tool_meta(vectorstore)




CODE_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are an AI teaching assistant specialized in evaluating Python code. Your task is to analyze the submitted code and evaluate it against the given practice question. Evaluate the code based on:
1. Correctness: Does it solve the problem posed in the practice question?
2. Syntax correctness
3. Functionality
4. Code style and best practices
5. Concept understanding

Practice Question: {practice_question}

Provide feedback in the following format:
- Overall assessment: (Correct/Partially Correct/Incorrect)
- Correctness: (Does the code correctly solve the problem posed in the practice question?)
- Comments: (Detailed explanation of what the code does well or where it falls short)
- Suggestions: (Specific ideas for improvement)"""),
    HumanMessagePromptTemplate.from_template("Here's the code to evaluate:\n\n{code}\n\nPlease provide your evaluation.")
])
# Define the practice question generation prompt
PRACTICE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""You are an AI teaching assistant specialized in creating Python programming practice questions. Your task is to generate a practice question based on the given topic or concept. The question should be:
1. Clear and concise
2. Appropriate for the user's skill level
3. Focused on a specific Python concept or skill
4. Accompanied by a brief explanation of what the question is testing

Provide the practice question in the following format:
Topic: (The main Python concept being tested)
Question: (The actual practice question)
Explanation: (A brief explanation of what the question is testing and why it's important)
Skill Level: (The skill level this question is appropriate for)
Justification: (Why this question is appropriate for the given skill level)"""),
    HumanMessagePromptTemplate.from_template("Please generate a practice question for the following topic: {topic}. The user's skill level is: {skill_level}.")
])







# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to get context for code evaluation
def get_code_context(code):
    retrieved_docs = retriever(code)
    return format_docs(retrieved_docs)



# Helper function to get context for practice questions
def get_question_context(input_dict):
    topic = input_dict["topic"]
    retrieved_docs = retriever(topic)
    return format_docs(retrieved_docs)

# Helper function to parse the AI's response
def parse_practice_question(response):
    lines = response.split('\n')
    parsed = {
        "topic": "",
        "question": "",
        "explanation": "",
        "skill_level": "",
        "justification": ""
    }
    current_key = None
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in parsed:
                parsed[key] = value
                current_key = key
            elif key == "skill level":  # Handle "Skill Level" separately
                parsed["skill_level"] = value
                current_key = "skill_level"
        elif current_key:
            parsed[current_key] += ' ' + line.strip()
    return parsed
def debug_input(x):
    print(f"Input to LLM: {x}")
    return x

def setup_sidebar():
    st.sidebar.title("Python Learning Assistant Configuration")

    # LLM selection for Code Evaluation
    code_eval_provider = st.sidebar.selectbox("AI for Code Evaluation", 
                                              list(LLMManager.get_provider_models().keys()))
    code_eval_model = st.sidebar.selectbox("Model for Code Evaluation", 
                                           LLMManager.get_models_for_provider(code_eval_provider))

    st.sidebar.write("---")

    # LLM selection for Practice Question Generation
    question_gen_provider = st.sidebar.selectbox("AI for Question Generation", 
                                                 list(LLMManager.get_provider_models().keys()))
    question_gen_model = st.sidebar.selectbox("Model for Question Generation", 
                                              LLMManager.get_models_for_provider(question_gen_provider))

    return code_eval_provider, code_eval_model, question_gen_provider, question_gen_model

# Streamlit UI components
st.title("Python Learning Assistant")

# Initialize session state
if 'code_input' not in st.session_state:
    st.session_state.code_input = ""
if 'code_evaluation' not in st.session_state:
    st.session_state.code_evaluation = ""
if 'topic_input' not in st.session_state:
    st.session_state.topic_input = ""
if 'skill_level' not in st.session_state:
    st.session_state.skill_level = "Beginner"
if 'practice_question' not in st.session_state:
    st.session_state.practice_question = ""
if 'current_practice_question' not in st.session_state:
    st.session_state.current_practice_question = None
if 'question_generated' not in st.session_state:
    st.session_state.question_generated = False

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Practice Questions", "Code Evaluation"])

# Add the new sidebar configuration
code_eval_provider, code_eval_model, question_gen_provider, question_gen_model = setup_sidebar()

# Load the LLMs after the sidebar setup
code_eval_llm = LLMManager.load_llm(code_eval_provider, code_eval_model)
question_gen_llm = LLMManager.load_llm(question_gen_provider, question_gen_model)

# Set up the chain for practice question generation
practice_question_chain = (
    RunnablePassthrough.assign(context=get_question_context)
    | PRACTICE_QUESTION_PROMPT
    | RunnableLambda(debug_input)
    | question_gen_llm
    | StrOutputParser()
)

# Set up the RAG chain for code evaluation
code_evaluation_chain = (
    RunnablePassthrough.assign(context=lambda x: get_code_context(x['code']))
    | CODE_EVALUATION_PROMPT
    | code_eval_llm
    | StrOutputParser()
)


if page == "Practice Questions":
    st.header("Generate Practice Questions")

    # Display current practice question if it exists
    if st.session_state.question_generated:
        st.subheader("Current Practice Question")
        for key, value in st.session_state.current_practice_question.items():
            if value:
                st.markdown(f"**{key.capitalize()}:** {value}")

    # Topic input
    st.session_state.topic_input = st.text_input("Enter the Python topic or concept you want to practice:", value=st.session_state.topic_input)

    # Skill level selection
    st.session_state.skill_level = st.select_slider("Select your skill level:", 
                                   options=["Beginner", "Intermediate", "Advanced"],
                                   value=st.session_state.skill_level)

    # Generate question button
    if st.button("Generate Practice Question"):
        if st.session_state.topic_input:
            with st.spinner("Generating practice question..."):
                raw_response = practice_question_chain.invoke({
                    "topic": st.session_state.topic_input, 
                    "skill_level": st.session_state.skill_level,
                    "context": get_question_context({"topic": st.session_state.topic_input})
                })
                st.session_state.practice_question = parse_practice_question(raw_response)
                st.session_state.current_practice_question = st.session_state.practice_question  # Store the current question
                st.session_state.question_generated = True  # Set the flag

                # Display raw response for debugging
                st.subheader("Raw AI Response")
                st.text(raw_response)

                # Display parsed response
                st.subheader("Parsed Practice Question")
                for key, value in st.session_state.practice_question.items():
                    if value:  # Only display non-empty fields
                        st.markdown(f"**{key.capitalize()}:** {value}")
                    else:
                        st.warning(f"Missing {key} in the generated question.")
        else:
            st.warning("Please enter a topic or concept.")
    

elif page == "Code Evaluation":
    st.header("Code Evaluation")
    
    # Display current practice question
    if st.session_state.question_generated:
        st.subheader("Current Practice Question")
        st.write(st.session_state.current_practice_question['question'])
    else:
        st.warning("No practice question generated yet. Please go to the Practice Questions page to generate a question first.")

    
    # Code input area
    code_editor_box = st_ace(language="python", theme="dracula",show_gutter=True,auto_update=True,wrap=True, value=st.session_state.code_input)
    
    # Custom CSS for button styling
    st.markdown("""
    <style>
    .stApp button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Evaluation button
    if st.button("Evaluate Code", key='custom_apply', type="primary"):
        if code_editor_box and st.session_state.question_generated:
            with st.spinner("Evaluating code..."):
                evaluation_input = {
                    "code": code_editor_box,
                    "practice_question": st.session_state.current_practice_question['question']
                }
                st.session_state.code_evaluation = code_evaluation_chain.invoke(evaluation_input)
            st.subheader("Evaluation Result")
            st.write(st.session_state.code_evaluation)
        elif not st.session_state.question_generated:
            st.warning("Please generate a practice question first.")
        else:
            st.warning("Please enter some code to evaluate.")
    st.session_state.code_input = code_editor_box
        
  

