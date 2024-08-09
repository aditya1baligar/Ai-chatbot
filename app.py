import streamlit as st
from streamlit_chat import message
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

import time

from dotenv import load_dotenv

load_dotenv()

# Load OpenAI API Key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("ProcurEngine Chatbot")

# Greeting message logic
if 'greeted' not in st.session_state:
    st.session_state['greeted'] = False

if not st.session_state['greeted']:
    st.write("Hello! Welcome to the ProcurEngine Chatbot. How can I assist you today?")
    st.session_state['greeted'] = True

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    """
    
    You are a Guide for the customers to navigate through the ProcurEngine software.
    Answer the questions based on the erocurement in such a way it should link in to the given context provided, also suggest what user can ask related to provided context and the question of the user.
    Please provide the most accurate response based on the question.
    If asked for the steps or configure, give a detailed step by step process for the question.
    Always provide all steps when the question is about configuration or steps.
    Just start with the answer and don't provide unnecessary information and if the information is not found in the context but is related to procurement then provide a basic knownledge about that question.
    Explain in detail for all the questions asked.
    Provide sufficient spaces to make the output readable.
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

def vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.text_loader_kwargs = {'autodetect_encoding': True}
    st.session_state.loader = DirectoryLoader("./procurEngine-data/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=st.session_state.text_loader_kwargs) # Data Ingestion
    st.session_state.docs = st.session_state.loader.load() # Document Loading
                                                                        
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Chunk Creation
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs) # Splitting
    
    # Initialize Chroma with proper tenant configuration
    try:
        st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        # st.write("Vector Store DB is ready")
    except ValueError as e:
        st.error(f"Error initializing Chroma: {str(e)}")

# Automatically perform vector embedding when the app loads
if "vectors" not in st.session_state:
    vector_embedding()

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

prompt1 = st.text_input("Enter your question")

# List of common questions button
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    button1 = st.button('How to configure Reverse Auction?')
    if button1:
        prompt1 = "How to configure Reverse Auction? Explain in detail."

with col2:
    button2 = st.button('How to register a vendor?')
    if button2:
        prompt1 = "How to register a vendor? Explain the methods in detail."

with col3:
    button3 = st.button('How to reach out to us?')
    if button3:
        prompt1 = "How to reach out to us?"

with col4:
    button4 = st.button('How to configure eRFQ?')
    if button4:
        prompt1 = "How to configure eRFQ? Explain in detail."
        
with col5:
    button5 = st.button('How to configure NFA?')
    if button5:
        prompt1 = "How to configure NFA? Explain in detail."
        
with col6:
    button6 = st.button('General rules of Reverse Auction')
    if button6:
        prompt1 = "What are the general rules of Reverse Auction? Explain in detail."

def get_complete_response(retrieval_chain, prompt1):
    response = retrieval_chain.invoke({"input": prompt1})
    if "Sorry! Information not Found!" not in response['answer']:
        # Check if all steps are included
        if "Step 11:" not in response['answer']:
            # Attempt to retrieve more context or rephrase the query
            additional_response = retrieval_chain.invoke({"input": "Provide all steps to configure reverse auction"})
            response['answer'] += additional_response['answer']
    return response

if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_k=10)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time:", time.process_time() - start)
    # Store the output 
    st.session_state.past.append(prompt1)
    st.session_state.generated.append(response['answer'])
    
    # Display response
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------------------")
