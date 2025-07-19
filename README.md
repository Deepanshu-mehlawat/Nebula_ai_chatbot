
# 📄 Enterprise Document Q&A with Multimodal RAG

This project is a powerful, locally-run Question-Answering system that leverages a Retrieval-Augmented Generation (RAG) architecture. It is designed to ingest and understand a variety of enterprise documents, including PDFs, spreadsheets, and emails.

A key feature is its **multimodal capability**: it not only processes text but can also analyze images within documents (like charts, diagrams, and figures), making them searchable through natural language queries. The system is powered by Google's Gemini family of models and is built to run easily on a local machine without complex local LLM setup.

---

## ✨ Key Features

- **Multi-Format Ingestion:** Process PDFs, Excel spreadsheets (`.xlsx`), and CSV files.
- **True Multimodal Understanding:** Extracts and analyzes both text and images from PDFs. Images are described using a vision-capable LLM, making their content searchable.
- **RAG-Powered Chatbot:** Ask questions in plain English and get answers generated from the content of your documents.
- **Accurate Citations:** Every answer includes references to the source document and page number from which the information was retrieved.
- **Interactive UI:** A simple and clean web interface built with Streamlit for easy interaction.
- **Local & Free:** Runs on your local machine using free-tier cloud APIs for the AI models, with no need for a dedicated GPU.

---

## 🛠️ Tech Stack

- **Backend & Orchestration:** Python, LangChain  
- **Document Processing:**
  - **Text (PDF):** `PyMuPDF` (`fitz`)
  - **Images (PDF):** `PyMuPDF` for extraction, `Pillow` for handling
  - **Spreadsheets:** `Pandas`
- **AI Models (via API):**
  - **Embeddings:** Google AI (`text-embedding-001`)
  - **Text Generation (LLM):** Google AI (`gemini-1.0-pro`)
  - **Image Description (Vision):** Google AI (`gemini-1.5-flash-latest`)
- **Vector Database:** Chroma DB (runs locally)
- **Frontend:** Streamlit

---

## 📁 Project Structure

```
doc-qa-rag/
│
├── .env                  # Stores secret API keys (DO NOT COMMIT)
├── .gitignore            # Specifies files for Git to ignore
├── requirements.txt      # List of all Python dependencies
├── process.py            # Script for data ingestion, processing, and indexing
├── app.py                # The main Streamlit chat application script
│
├── data/                 # Place all your source documents here
│   └── example.pdf
│   └── financial_report.xlsx
│
├── images/               # Temp folder for extracted images (auto-created)
│
└── chroma_db/            # Local vector database storage (auto-created)
```

---

## 🚀 Setup and Installation

### Step 0: Prerequisites

1. **Python:** Ensure you have Python 3.9+ installed.  
2. **Git:** Required for cloning the repository.  
3. **Conda (Recommended):** Using a Conda environment is highly recommended to manage dependencies cleanly.  
4. **Tesseract OCR Engine:** Some libraries may probe for it even if unused directly.  
   - Download and install from the [official Tesseract repository for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
   - **Important:** Add Tesseract to your system's `PATH`.

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd doc-qa-rag
```

### Step 2: Set Up the Python Environment

> We recommend using Conda to avoid conflicts with other projects.

```bash
# Create a new Conda environment named 'doc_qa' with Python 3.10
conda create -n doc_qa python=3.10 -y

# Activate the new environment
conda activate doc_qa
```

### Step 3: Install Dependencies

Install all Python libraries:

```bash
pip install -r requirements.txt
```

### Step 4: Configure Your API Key

1. **Get a Google AI API Key:**
   - Visit [Google AI Studio](https://aistudio.google.com/).
   - Sign in and create a new API key.

2. **Create the `.env` file:**

```env
GOOGLE_API_KEY="your_api_key_goes_here"
```

> The `.gitignore` file already ensures this stays private.

---

## 💡 How to Use the System

The system works in two stages: first, you process your documents to create a knowledge base, and second, you run the chat interface to ask questions.

### Stage 1: Process and Index Your Documents

1. **Add Your Documents:**  
   Place your PDFs, `.xlsx`, and `.csv` files into the `/data` folder.

2. **Run the Processing Script:**  
   With the `doc_qa` environment activated, run:

```bash
python process.py
```

This script will:
- Read each file in `/data`
- Extract text and images from PDFs
- Send images to the Gemini Vision API for descriptions
- Generate embeddings from both text and image content
- Store them in a local vector database in `/chroma_db`

> Only rerun this when documents change.

---

### Stage 2: Launch the Chatbot and Ask Questions

1. **Run the Streamlit App:**

```bash
streamlit run app.py
```

2. **Interact:**
   A web browser will open the chat UI. Ask questions based on your documents—answers include citations with file name and page number.

---
