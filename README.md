# AI-Agent-Dashboard


## Project Overview
This project is designed to showcase an AI-powered agent that automates web searches and retrieves structured information from public data. Users can upload a CSV file or connect to a Google Sheet, define search queries, and view/download the results.

## Features
- **CSV and Google Sheets Integration**: Upload or connect to data sources easily.
- **Dynamic Querying**: Use custom queries with placeholders like `{entity}`.
- **Automated Web Search**: Retrieve structured results using APIs.
- **LLM Integration**: Process search results using Groq or OpenAI.
- **Data Output**: Download results as CSV or update directly in Google Sheets.

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Data Handling**: Pandas, Google Sheets API
- **Web Search**: SerpAPI
- **LLM**: Groq API via LangChain
- **Others**: Tenacity for retry logic, dotenv for environment variable management.


