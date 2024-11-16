import streamlit as st
import pandas as pd
import os
import sys
import requests
import json
import re
import time
import logging
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import h2.connection
from groq import Groq
import gspread
import random
from oauth2client.service_account import ServiceAccountCredentials
from logging.handlers import RotatingFileHandler


from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
st.header("AI Agent Dashboard")

        


st.write("This app allows you to upload a CSV file or connect to Google Sheets, preview the data, and choose an entity column.")

# Step 1: File Upload
st.header("Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data)
    st.write("---")  # Divider for visual separation

# Step 2: Google Sheets Connection
st.header("Connect to Google Sheets")
google_sheet_url = st.text_input("Enter Google Sheets URL")

if google_sheet_url:
    try:
        # Load Google Sheets credentials from service account file
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON"),
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
        )

        # Extract the Sheet ID from the URL
        sheet_id = google_sheet_url.split("/d/")[1].split("/")[0]
        service = build("sheets", "v4", credentials=creds)

        # Access the first sheet's data as a DataFrame
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=sheet_id, range="Sheet1").execute()
        values = result.get("values", [])

        if values:
            # Convert Google Sheets data to a DataFrame
            data = pd.DataFrame(values[1:], columns=values[0])  # Assumes first row is the header
            st.write("Preview of Google Sheets Data:")
            st.dataframe(data)
        else:
            st.write("No data found in the Google Sheet.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    st.write("---")  # Divider for visual separation

# Add the Dynamic Query Input section to the dashboard
st.header("Define Your Query")

# Query input with a default example to guide users
query_template = st.text_input(
    "Enter your query (e.g., 'Get the email address of {company}')"
)

# Placeholder dropdown for selecting main column (from Step 1)
# Ensure that 'data' is loaded either from CSV or Google Sheets
if 'data' in locals():
    column = st.selectbox("Choose the primary column for entity search", data.columns)

    # Display the query with placeholders replaced by actual data for a preview
    if query_template and column:
        # Take the first few items in the selected column to show sample queries
        sample_entities = data[column].head(3)  # Using first 3 rows for preview
        st.write("Sample Queries:")

        # Generate and display sample queries
        for entity in sample_entities:
            sample_query = query_template.replace("{company}", str(entity))
            st.write(f"- {sample_query}")
    else:
        st.write("Please upload a CSV or connect to a Google Sheet to proceed.")
# Load environment variables
load_dotenv()

# SerpAPI setup (or ScraperAPI if you prefer)
SERP_API_KEY = os.getenv("SERP_API_KEY")
SERP_API_URL = "https://serpapi.com/search"

# Function to perform a web search using SerpAPI
def perform_web_search(query):
    # Define API request parameters
    params = {
        "engine": "google",  # Use Google search engine
        "q": query,          # The search query
        "api_key": SERP_API_KEY
    }
    # Make a GET request to SerpAPI
    response = requests.get(SERP_API_URL, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        search_results = response.json()
        return search_results.get("organic_results", [])  # Organic search results
    else:
        return None

# Streamlit section to execute search queries
st.title("Perform Web Search")

if query_template and 'data' in locals() and column:
    st.write("Fetching information for each entity...")

    # Create an empty dictionary to store search results for each entity
    search_results = {}

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Get the total number of entities for the progress bar
    total = len(data[column])

    # Iterate over each entity in the selected column and perform the search
    for idx, entity in enumerate(data[column]):
        # Replace placeholder with the actual entity in the query template
        modified_query = query_template.replace("{company}", str(entity))
        
        # Perform the web search
        results = perform_web_search(modified_query)

        # Store results in the dictionary with a fallback for missing data
        search_results[entity] = []
        if results:
            for result in results[:3]:  # Limit to first 3 results
                search_results[entity].append({
                    "Title": result.get("title", "N/A"),
                    "Link": result.get("link", "N/A"),
                    "Snippet": result.get("snippet", "N/A")
                })
        else:
            search_results[entity].append({
                "Title": "No results",
                "Link": "N/A",
                "Snippet": "N/A"
            })
        
        # Update progress bar
        progress_bar.progress((idx + 1) / total)

    # Store results in session state for later use
    st.session_state.search_results = search_results

    # Display the search results in a structured format
    st.header("Extracted Information")
    # Flatten the results for DataFrame display
    flat_results = []
    for entity, results in search_results.items():
        for result in results:
            flat_results.append({
                "Entity": entity,
                **result  # Unpacking Title, Link, Snippet
            })
    
    
    df_results = pd.DataFrame(flat_results)
    st.dataframe(df_results)

    # Provide download button for the results
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False),
        file_name="search_results.csv",
        mime="text/csv"
    )
    def fetch_search_results(query):
        try:
            # Define API request parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERP_API_KEY
            }
            # Make a GET request to SerpAPI
            response = requests.get(SERP_API_URL, params=params)
            response.raise_for_status()  # Raise an error for HTTP failures
    
            # Parse the JSON response
            search_results = response.json()
            return search_results.get("organic_results", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Web search failed: {e}") from e
            
        

    # Update Google Sheet

    update_button = st.button("Update Google Sheet")
    if update_button:
            try:
                # Ensure credentials and service are already set up
                sheet = service.spreadsheets()
                # Prepare data for Google Sheet update
                values = [df_results.columns.tolist()] + df_results.values.tolist()
                
                # Write the results back to the sheet in a new tab or specified range
                update_range = "Results!A1"  # Create a new tab named 'Results' and write data
                body = {
                    "range": update_range,
                    "majorDimension": "ROWS",
                    "values": values
                }
                
                # Perform the update
                sheet.values().update(
                    spreadsheetId=sheet_id,
                    range=update_range,
                    valueInputOption="RAW",
                    body=body
                ).execute()
                
                st.success("Google Sheet updated successfully!")
            except Exception as e:
                st.error(f"Failed to update Google Sheet: {e}")
else:
    st.write("Please define a query template and select a column to proceed.")


#Groq data extraction
def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key =  "YOUR_GROQ_API"

    if 'search_results' not in st.session_state:
        st.session_state.search_results = {}  # Initialize empty dict for search results
    
    
    # Example of processing results and storing them in session_state
    entity = "Sample Entity"  # Replace this with actual entity processing logic
    response = {
        "success": True,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "search_query": f"Search query for {entity}",
        "results": "Sample processed data",
        "raw_search": {"sample_key": "sample_value"}
    }
    
    # Store the response in session state under the respective entity
    st.session_state.search_results[entity] = response


    
   

    # The title and greeting message of the Streamlit application
    st.title("Chat with Groq!")

    # Add customization options to the sidebar
    st.sidebar.title('Groq Customization')
    system_prompt = st.sidebar.text_input("System prompt:",value="Enter your query:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Search Your Query:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input':message['human']},
                {'output':message['AI']}
                )


    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )


    # If the user has asked a question,
    if user_question:

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
       
            memory=memory,
              # The conversational memory object that stores and manages the conversation history.
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human':user_question,'AI':response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

    
  

if __name__ == "__main__":
    main()

