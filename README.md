```markdown
# AI Troubleshooting Assistance - Proof of Concept

## Overview

This repository contains a proof of concept (PoC) for an AI-driven troubleshooting assistant designed for locomotive systems. It uses **Streamlit** for the frontend interface, **Langchain** for language models and document retrieval, and **GROQ API** for efficient model execution. The application leverages **PDF documents** containing technical data to generate detailed, context-aware troubleshooting instructions based on user inputs.

## Key Features

- **Context-Aware Troubleshooting:** Uses a machine learning model to provide answers to specific locomotive issues based on the context derived from technical documentation.
- **Streamlit UI:** Simple, interactive interface for users to enter queries and retrieve solutions.
- **PDF Document Embedding:** Loads and processes PDF documents containing technical data for efficient retrieval using FAISS vector store.
- **Custom AI Prompts:** Langchain-based custom prompts are used to ensure the AI provides the most relevant troubleshooting responses.

## Setup

### Prerequisites

Make sure you have **Python 3.7+** installed and **Streamlit** is available.

### 1. Clone the Repository:

```bash
git clone https://github.com/your-username/ai-troubleshooting-system.git
cd ai-troubleshooting-system
```

### 2. Install Dependencies:

Install the required libraries using the following:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables:

- Create a `.env` file in the root directory and add your **GROQ API key**:
  ```
  GROQ_API_KEY=your-groq-api-key
  ```
- Alternatively, if you're using **Streamlit Cloud**, you can configure the key via **Streamlit Secrets**.

### 4. Start the Application:

Run the Streamlit app with:

```bash
streamlit run app.py
```

## Application Workflow

1. **Load Documents:** The system loads and processes PDFs from the `input_sample/` directory on startup. These documents are embedded and indexed for efficient retrieval.
2. **User Input:** Users can input a query or issue description (e.g., "Locomotive DE24333 has electrical issue with control system").
3. **Document Retrieval & AI Answering:** The system retrieves relevant documents, processes the query, and generates a context-aware troubleshooting response.
4. **Display Results:** The answer, along with the retrieved document context, is displayed on the Streamlit interface.

## Troubleshooting

- **No PDFs Found:** Ensure that the `input_sample/` directory contains PDF files. If this directory is empty or the files are missing, the app will display an error message indicating that no PDFs were found.
- **Embedding Issues:** If document embedding fails, verify that the PDFs are valid and can be processed by the **PyMuPDF** loader. Additionally, ensure that the **GROQ API** key is correctly configured.
- **Slow Response:** The system may take time to generate answers depending on the complexity of the query and the size of the documents being processed.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Credits

- **Langchain:** For providing powerful tools to chain language models and document processing.
- **Streamlit:** For making the web interface quick and interactive.
- **Hugging Face:** For providing transformer-based embeddings for document similarity matching.
- **PyMuPDF:** For handling PDF document extraction.

```

This `README.md` includes all the necessary setup, features, and troubleshooting steps to ensure that the user can easily understand, install, and use the project. Feel free to modify the URLs or sections if you need them more specific to your project.
