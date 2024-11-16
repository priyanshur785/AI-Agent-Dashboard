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

# **AI Agent for Data-Driven Information Retrieval**

## **Project Description**
This project automates the retrieval of information about entities from a dataset using advanced AI and API integrations. It allows users to:
- Upload a CSV file or connect to a Google Sheet.
- Define custom prompts for querying specific information about entities in the dataset.
- Perform automated web searches and extract data using a Large Language Model (LLM).
- View the extracted information in a user-friendly dashboard and export it as a CSV or update the connected Google Sheet.

Key technologies used include Python, Streamlit for the user interface, SerpAPI for web searches, and OpenAI’s GPT API for data processing.

---

## **Setup Instructions**

### **1. Clone the Repository**
-Clone the project repository to your local system
### **2. Install Dependencies**
-Ensure Python 3.9 or higher is installed
### **2. Configure Environment Variables**
-Create a .env file in the root of the project and add the following:
  SERPAPI_KEY=<Your SerpAPI Key>
  GOOGLE_SHEET_CREDENTIALS=<Path to your Google Sheets credentials JSON file>
### **2. Run the Application**
-Start the Streamlit dashboard:
 streamlit run dashboard.py

## **Usage Guide**

### **1. Uploading a Dataset**
- Launch the application using:
  streamlit run dashboard.py
- Navigate to the dashboard interface.
- Use the "Browse" button to upload a CSV file. Alternatively, connect to a Google Sheet by 
  entering the required credentials and authenticating access.
-The uploaded data will be previewed in a table. Select the primary column you want to query 
 (e.g., Company Names).
### **2. Defining a Query**
- Enter a custom query in the input box. Use curly braces {} as placeholders for dynamic replacement based on the column entities.
Examples:
Get the email address of {company}.
Find the headquarters location of {organization}.
### **3. Processing the Data**
-Click the "Search" button to start processing.
  The system will perform web searches using APIs and retrieve relevant results.
  Results are processed by the Groq API to extract specific information based on your query.
 ### **4. Viewing and Exporting Results**
 -The results are displayed in a clean table format on the dashboard.
  Export options:
     Download CSV: Save the results locally as a CSV file.
     Update Google Sheet: Push the extracted results directly to the connected Google Sheet.

## **API Keys and Environment Variables**

### **API Keys**
To enable the core functionalities of this project, the following API keys are required:

1. **SerpAPI**  
   - Used for performing automated web searches to retrieve relevant information.
   - Obtain your API key by signing up at [SerpAPI](https://serpapi.com/).

2. **Groq API**  
   - Used as the LLM for processing search results and extracting specific information.
   - Retrieve your API key by signing up on [Groq's platform](https://groq.com/).

3. **Google Sheets API**  
   - Used for real-time interaction with Google Sheets to fetch and update data.
   - Follow these steps to enable and set up the Google Sheets API:
     - Visit the [Google Cloud Console](https://console.cloud.google.com/).
     - Enable the **Google Sheets API** for your project.
     - Generate and download the credentials JSON file for your project.

## Loom Video Link
 Link: https://www.loom.com/share/3b770d83f301473b8992f18e9a9488cf?sid=ea2a0fae-4d01-4c13-8c02-14156a52b3b1


