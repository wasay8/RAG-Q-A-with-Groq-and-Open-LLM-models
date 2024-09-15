# RAG Formal Method Q&A with Groq and Open Source LLM Models

This Streamlit app allows users to perform question and answer (Q&A) operations on research papers using various LLM models and vector embeddings. The app leverages Groq AI and Hugging Face models for processing and querying document contexts.

## Requirements

- Python 3.7+
- Streamlit
- LangChain (including `langchain_groq`, `langchain_openai`, `langchain_community`, etc.)
- OpenAI
- `python-dotenv` for environment variable management

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a `.env` file in the same directory as `app.py` with the following content:

   ```env
   HF_TOKEN=your_hugging_face_token
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. **Prepare Documents:**

   Create a `Documents` folder in the same directory as `app.py` and place all PDF files you want to use as context for Q&A.

2. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

3. **Interact with the App:**

   - Use the sidebar to input your Groq API key and Hugging Face API key.
   - Click on "Document Embedding" to load and process the documents.
   - Select an LLM model from the dropdown menu and set the temperature for the model.
   - Enter your query in the text input box and view the response.

## Notes

- Ensure that the paths and API keys are correctly set up in the `.env` file.
- Adjust the paths and settings according to your local setup if needed.

## Acknowledgement

Special thanks to Krish Naik’s for guidance.

## Troubleshooting

- If no documents are loaded, verify the file path and contents in the `Documents` folder.
- Check your API keys and ensure they are correct.
- For any errors, refer to the error messages displayed in the app or consult the LangChain documentation for additional guidance.

Feel free to contribute or open issues if you encounter any problems or have suggestions for improvements!






