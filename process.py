# File: process.py (Final, Simplified Version)

import os
import pandas as pd
import email
from email import policy
from dotenv import load_dotenv
import fitz  # PyMuPDF

# --- CORRECTED IMPORTS ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# --- END OF CORRECTIONS ---

# Load environment variables from .env
load_dotenv() 

# --- Main Processing Logic ---
DATA_PATH = "data/"
all_texts = []
all_metadatas = []

print("Starting document processing...")
if not os.path.isdir(DATA_PATH):
    print(f"Error: Data directory '{DATA_PATH}' not found. Please create it and add your documents.")
    exit()

for filename in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.isfile(file_path):
        continue
        
    print(f"-> Processing '{filename}'...")
    
    try:
        # --- MODIFICATION: Replaced 'unstructured' with 'PyMuPDF' for reliability ---
        if filename.endswith(".pdf"):
            doc_texts = []
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    if page_text: # Only add pages that have text
                        all_texts.append(page_text)
                        # Add page number to metadata for better citation
                        all_metadatas.append({"source": filename, "page": page_num + 1})
            print(f"   Successfully extracted text from {len(doc)} pages.")
        # --- END OF MODIFICATION ---
        
        elif filename.endswith((".xlsx", ".csv")):
            df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path)
            for index, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                all_texts.append(f"In {filename}, row {index+1} contains: {row_text}")
                all_metadatas.append({"source": filename})
            print(f"   Successfully processed {len(df)} rows.")
            
        elif filename.endswith(".eml"):
            # Email parsing logic remains the same
            with open(file_path, 'r', encoding='utf-8') as f:
                msg = email.message_from_file(f, policy=policy.default)
            subject = msg.get('subject', 'No Subject')
            sender = msg.get('from', 'Unknown Sender')
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors='ignore')
            all_texts.append(f"Email from {sender} with subject '{subject}': {body}")
            all_metadatas.append({"source": filename})
            print("   Successfully processed email.")
            
    except Exception as e:
        print(f"  !! An error occurred while processing {filename}: {e}")

print(f"\nTotal text chunks extracted across all files: {len(all_texts)}")

if not all_texts:
    print("\nWarning: No text was extracted. Please check your files.")
else:
    print("\nInitializing embedding model...")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Creating and persisting vector database... (This may take a moment)")
        vector_db = Chroma.from_texts(
            texts=all_texts,
            embedding=embedding_model,
            metadatas=all_metadatas,
            persist_directory="./chroma_db"
        )
        print("\nVector database created and persisted successfully in the 'chroma_db' folder!")
    except Exception as e:
        print(f"\n!! An error occurred during embedding or database creation: {e}")
