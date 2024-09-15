# RAG-Q-A-with-Groq-and-Open-LLM-models


This Streamlit application allows users to interact with research papers using various LLM (Large Language Models) and vector databases. The application leverages Groq AI for querying and integrates different embeddings and retrieval models to provide accurate answers based on provided contexts.

## Features

- **Document Embedding**: Load and process PDF documents to create a vector database.
- **Interactive Query**: Enter queries and receive answers based on the loaded documents.
- **Model Selection**: Choose between different LLM models for querying.
- **Temperature Control**: Adjust the temperature setting to control the creativity of the LLM responses.
- **Document Similarity Search**: View similar documents related to the query.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:
   Create a `.env` file in the project directory and add your API keys:
   ```
   HF_TOKEN=your_hugging_face_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the App**:
   - **Enter your Groq API Key and Hugging Face API Key** in the sidebar.
   - **Upload PDF Documents** using the "Document Embedding" button.
   - **Enter your Query** and select the LLM model from the sidebar.
   - **Adjust Temperature** to control the response creativity.
   - **View Responses and Document Similarities** in the app.

## Code Overview

- **Sidebar Settings**: Allows users to input API keys and configure settings.
- **Vector Embedding Creation**: Loads and processes PDF documents to create embeddings and a FAISS vector store.
- **LLM Model Selection**: Allows selection of different LLM models and adjusts temperature for querying.
- **Query Processing**: Uses the selected model to answer user queries based on the provided documents.
- **Document Similarity Search**: Displays similar documents related to the user’s query.

## Credits

Special thanks to Krish Naik’s for this application guidance.

