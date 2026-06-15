!pip install langchain-google-genai -qqq
!pip install --upgrade langchain-google-genai langchain-core -qqq
!pip install pinecone-client langchain langchain-community langchain-google-genai pypdf -qqq
!pip uninstall -y pinecone-client pinecone -qqq
!pip install pinecone langchain langchain-community langchain-google-genai pypdf -qqq

import os

# Injecting keys directly into the environment state
os.environ['GOOGLE_API_KEY'] = 'AQ.Ab8RN6LdQIhYYVh50FyPPZ0zF0LwRvHCAniM1owQWFnRBXHueQ'
os.environ['PINECONE_API_KEY'] = 'pcsk_5YUQ8L_7NWNfC4219AfEjqygUGFJcreC9MWPuqbFUMUPii4GCi48n7AP5Tasc4MHuaDgk9'
os.environ['PINECONE_INDEX_NAME'] = 'pureplate-rag-index'

"""
ingest.py - Document Ingestion and Indexing Pipeline
Author: PurePlate AgroFood AI Team
Description: Loads standard text PDFs, splits content into semantic chunks,
             generates Google Gemini embeddings efficiently via batched requests,
             forcing a 768 dimension output layout to match your index constraints.
"""

import os
import sys
import time
import warnings

# Suppress deep learning implementation framework environment and library deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_ingestion(data_dir: str = "./data"):
    """
    Executes the end-to-end extraction, splitting, vector embedding generation, 
    and native vector database ingestion deployment pipeline.
    """
    
    # 1. Safe Conditional Library Dynamic Loading
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from pinecone import Pinecone, ServerlessSpec
    except (ModuleNotFoundError, Exception) as e:
        print(f"\n[!] DEPENDENCY ERROR: Problem loading required library: {str(e)}")
        print("Please run the following environment clean-up line inside an active notebook cell:")
        print("!pip uninstall -y pinecone-client pinecone && pip install pinecone langchain langchain-community pypdf")
        return  # Safe return preventing IPython trace dump failures

    # Fallback sequence handling to dynamically locate Google GenAI Embeddings
    GoogleEmbeddingsClass = None
    try:
        from langchain_google_genai import GoogleGenAIEmbeddings
        GoogleEmbeddingsClass = GoogleGenAIEmbeddings
    except Exception:
        try:
            from langchain_google_genai.embeddings import GoogleGenAIEmbeddings
            GoogleEmbeddingsClass = GoogleGenAIEmbeddings
        except Exception:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                GoogleEmbeddingsClass = GoogleGenerativeAIEmbeddings
            except Exception as e:
                print(f"\n[!] CRITICAL IMPORT ERROR: Google GenAI Embedding modules missing. {str(e)}")
                print("Please execute: !pip install langchain-google-genai")
                return

    # Sourcing Environment configurations securely (Zero hardcoded secrets)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pureplate-rag-index")

    # Validate that necessary tokens are initialized before execution
    missing_keys = []
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not PINECONE_API_KEY:
        missing_keys.append("PINECONE_API_KEY")
        
    if missing_keys:
        print(f"\n[!] CONFIGURATION ERROR: Missing environment variables: {', '.join(missing_keys)}")
        print("Please set them up in your notebook workspace before running:")
        print("import os")
        print("os.environ['GOOGLE_API_KEY'] = 'your_key_here'")
        print("os.environ['PINECONE_API_KEY'] = 'your_key_here'")
        return

    print(f"\n[*] Starting Document Ingestion from local directory: '{data_dir}'...")
    
    # 2. Document Loading Lifecycle
    if not os.path.exists(data_dir):
        print(f"[-] Directory Absence: Target folder '{data_dir}' missing. Creating directory...")
        os.makedirs(data_dir)
        print(f"[!] Target folder created. Please place your PurePlate PDF files inside '{data_dir}' and re-run.")
        return
        
    try:
        loader = PyPDFDirectoryLoader(data_dir)
        documents = loader.load()
        print(f"[+] Document loading successful. Extracted {len(documents)} raw source pages.")
    except Exception as e:
        print(f"[-] Exception encountered during PDF Extraction Loop: {str(e)}")
        return

    if not documents:
        print("[-] Ingestion Halt: No valid PDF text records extracted from the target directory.")
        return

    # 3. Semantic Structural Character Chunk Splitting
    print("[*] Partitioning raw text layouts into semantic segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[+] Segment mapping finalized. Formed {len(chunks)} contextual text chunks.")

    # 4. Model Engine Generation Setup
    print(f"[*] Initializing embedding client using: {GoogleEmbeddingsClass.__name__}")
    try:
        # Base constructor config purely mapping model and API credentials
        embeddings_model = GoogleEmbeddingsClass(
            model="gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        print(f"[-] Google embedding instantiation failure: {str(e)}")
        return

    # 5. Programmatic Database Infrastructure Setup
    print("[*] Auditing Pinecone target workspace cluster...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        active_indexes = [idx.name for idx in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in active_indexes:
            print(f"[*] Target index '{PINECONE_INDEX_NAME}' not found. Provisioning index architecture...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,  # Strictly locked to align with target constraints
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"[+] Index '{PINECONE_INDEX_NAME}' successfully provisioned on Serverless AWS structure.")
            time.sleep(10)  # Propagation buffer for Serverless AWS DNS routing configurations
        else:
            print(f"[+] Valid connection established to preexisting Pinecone index: '{PINECONE_INDEX_NAME}'")
            
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"[-] Infrastructure Provisioning Exception encountered: {str(e)}")
        return

    # 6. Native Vector Store Population (Optimized Batch Vector Upsert)
    print(f"[*] Extracting embeddings and upserting {len(chunks)} items natively to Pinecone...")
    try:
        batch_size = 100
        
        for i in range(0, len(chunks), batch_size):
            # Isolate a segment block slice of 100 items max
            chunk_batch = chunks[i : i + batch_size]
            
            # Map chunk blocks into plain string lists for the API payload
            texts_to_embed = [chunk.page_content for chunk in chunk_batch]
            
            # FIXED: Explicitly supply output_dimensionality=768 here to bypass the 3072 default layout value
            vectors = embeddings_model.embed_documents(
                texts_to_embed, 
                output_dimensionality=768
            )
            
            upsert_data = []
            for j, chunk in enumerate(chunk_batch):
                global_index = i + j
                metadata = {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Unknown"),
                    "page": chunk.metadata.get("page", 0)
                }
                upsert_data.append((f"chunk_{global_index}", vectors[j], metadata))
            
            # Safely commit up to 100 vector payloads simultaneously to Pinecone
            index.upsert(vectors=upsert_data)
            print(f"[+] Successfully processed and indexed chunks {i} to {i + len(chunk_batch) - 1}")
                
        print("\n[+++] SUCCESS: Ingestion pipeline completed. Knowledge base initialized successfully!")
    except Exception as e:
        print(f"[-] Fatal Vector Upsert Failure: {str(e)}")

if __name__ == "__main__":
    # Dynamically handle workspace paths when running inside IPython interactive environments
    if len(sys.argv) > 1 and sys.argv[1] != "-f":
        target_folder = sys.argv[1]
    else:
        target_folder = "./data"
        
    run_ingestion(data_dir=target_folder)
