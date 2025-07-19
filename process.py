# File: process.py (FINAL MULTIMODAL VERSION)

import os
import pandas as pd
import email
from email import policy
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# --- CORRECTED IMPORTS ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# --- END OF CORRECTIONS ---

# Load environment variables and configure the Google API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# --- NEW: Helper function to describe an image using Gemini ---
def describe_image(image_path):
    """
    Uses a vision-capable Gemini model to generate a text description of an image.
    """
    try:
        # We use gemini-1.5-flash-latest as it's fast and powerful for multimodal tasks
        vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(image_path)
        
        # The prompt asks the model to be descriptive, which is great for RAG
        prompt = (
            "Describe this image in detail. If it contains a chart, graph, or table, "
            "extract the key information, data points, and conclusions. If it contains text, "
            "transcribe it. This description will be used for a question-answering system."
        )
        
        response = vision_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"  !! Could not describe image {image_path}: {e}")
        return None
# --- END NEW ---

# --- Main Processing Logic ---
DATA_PATH = "data/"
IMAGE_OUTPUT_PATH = "images/" # Folder to temporarily store extracted images
if not os.path.exists(IMAGE_OUTPUT_PATH):
    os.makedirs(IMAGE_OUTPUT_PATH)

all_texts = []
all_metadatas = []

print("Starting document processing...")
if not os.path.isdir(DATA_PATH):
    print(f"Error: Data directory '{DATA_PATH}' not found.")
    exit()

for filename in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.isfile(file_path):
        continue
        
    print(f"-> Processing '{filename}'...")
    
    try:
        if filename.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    # 1. Process regular text on the page
                    page_text = page.get_text("text")
                    if page_text:
                        all_texts.append(page_text)
                        all_metadatas.append({"source": filename, "page": page_num + 1, "type": "text"})

                    # 2. --- NEW: Extract and process images on the page ---
                    images = page.get_images(full=True)
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save the image temporarily
                        image_filename = f"{os.path.splitext(filename)[0]}_p{page_num+1}_img{img_index}.{image_ext}"
                        image_path = os.path.join(IMAGE_OUTPUT_PATH, image_filename)
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Get a text description of the image
                        print(f"   -> Describing image {image_filename}...")
                        description = describe_image(image_path)
                        if description:
                            all_texts.append(description)
                            all_metadatas.append({"source": filename, "page": page_num + 1, "type": "image"})
            print(f"   Successfully processed {len(doc)} pages (including text and images).")
            # --- END NEW ---

        elif filename.endswith((".xlsx", ".csv")):
            # This part remains the same
            df = pd.read_excel(file_path) if filename.endswith(".xlsx") else pd.read_csv(file_path)
            for index, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                all_texts.append(f"In {filename}, row {index+1} contains: {row_text}")
                all_metadatas.append({"source": filename})
            print(f"   Successfully processed {len(df)} rows.")

        # (Email processing can be added here if needed)

    except Exception as e:
        print(f"  !! An error occurred while processing {filename}: {e}")

print(f"\nTotal text and image description chunks extracted: {len(all_texts)}")

if not all_texts:
    print("\nWarning: No text was extracted. Please check your files.")
else:
    print("\nInitializing embedding model...")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Creating and persisting vector database... (This may take a moment)")
        
        # Before creating, it's good practice to clear the old DB
        # if os.path.exists("./chroma_db"):
        #     import shutil
        #     shutil.rmtree("./chroma_db")

        vector_db = Chroma.from_texts(
            texts=all_texts,
            embedding=embedding_model,
            metadatas=all_metadatas,
            persist_directory="./chroma_db"
        )
        print("\nVector database created and persisted successfully!")
    except Exception as e:
        print(f"\n!! An error occurred during embedding or database creation: {e}")
