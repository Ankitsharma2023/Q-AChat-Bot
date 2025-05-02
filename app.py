import streamlit as st
import openai 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os 


langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A ChatBot using Different LLMs"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions about the content of the document."),
        ("user", "Question:{question}"),
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

def generate_response1(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("Q&A ChatBot using Different LLMs")
st.sidebar.title("Settings")

model_source = st.sidebar.selectbox("Select Model Source", ["OpenAI", "Ollama"])

if model_source == "OpenAI":
    llm = st.sidebar.selectbox("Select LLM", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
else:
    llm = st.sidebar.selectbox("Select Ollama Model", ["llama2", "mistral", "gemma", "phi3"])

temperature = st.sidebar.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Select Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask me anything!")
user_input = st.text_input("You:")

if user_input:
    if model_source == "OpenAI":
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set it in Streamlit Secrets.")
        else:
            response = generate_response(user_input, openai_api_key, llm, temperature, max_tokens)
            st.write(response)
    else:
        response = generate_response1(user_input, llm, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please enter a question to get a response.")
