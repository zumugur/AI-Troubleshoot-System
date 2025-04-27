AI Troubleshooting System PoC using GROQ and Langchain. This application assists users in troubleshooting locomotive issues by leveraging technical documents, machine learning models, and context-aware AI to provide accurate solutions.

Updated README:
markdown
Copy
Edit
# AI Troubleshooting Assistance - Proof of Concept

## Overview

This repository contains a proof of concept (PoC) for an AI-driven troubleshooting assistant designed for locomotive systems. It uses **Streamlit** for the frontend interface, **Langchain** for language models and document retrieval, and **GROQ API** for efficient model execution. The application leverages **PDF documents** containing technical data to generate detailed, context-aware troubleshooting instructions based on user inputs.

## Key Features

- **Context-Aware Troubleshooting:** Uses a machine learning model to provide answers to specific locomotive issues based on the context derived from technical documentation.
- **Streamlit UI:** Simple, interactive interface for users to enter queries and retrieve solutions.
- **PDF Document Embedding:** Loads and processes PDF documents containing technical data for efficient retrieval using FAISS vector store.
- **Custom AI Prompts:** Langchain-based custom prompts are used to ensure the AI provides the most relevant troubleshooting responses.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/ai-troubleshooting-system.git
   cd ai-troubleshooting-system
Install Dependencies: Make sure you have Python 3.7+ installed. Then, install the required libraries.

bash
Copy
Edit
pip install -r requirements.txt
Set Up Environment Variables:

Create a .env file in the root directory and add your GROQ API key:

ini
Copy
Edit
GROQ_API_KEY=your-groq-api-key
Alternatively, you can configure the key via Streamlit Secrets (if using Streamlit cloud).

Start the Application: Run the Streamlit app with:

bash
Copy
Edit
streamlit run app.py
Application Workflow
Step 1: Load Documents: The system loads and processes PDFs from the input_sample/ directory on startup. These documents are embedded and indexed for efficient retrieval.

Step 2: User Input: Users can input a query or issue description (e.g., "Locomotive DE24333 has electrical issue with control system").

Step 3: Document Retrieval & AI Answering: The system retrieves relevant documents, processes the query, and generates a context-aware troubleshooting response.

Step 4: Display Results: The answer, along with the retrieved document context, is displayed on the Streamlit interface.

Troubleshooting
No PDFs Found: Ensure that the input_sample/ directory contains PDF files. If this directory is empty or the files are missing, the app will display an error.

Embedding Issues: If document embedding fails, verify that the PDFs are valid and can be processed by the PyMuPDF loader.

License
This project is licensed under the MIT License - see the LICENSE file for details.

yaml
Copy
Edit

---

### Key Points:
- The description highlights the key purpose of the app: troubleshooting locomotive systems using AI and technical documentation.
- The README covers setup instructions, usage flow, and possible issues like missing PDFs or failed embeddings.






