import os
import glob
import argparse
import json
import re
import uuid 
from typing import List, Dict, Tuple, Optional, Set

import torch 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder 
from transformers import pipeline

import nltk
from nltk.tokenize import sent_tokenize

# --- Configuration for Advanced RAG ---
DOC_FOLDER = "company_docs"
INDEX_FILE = "faiss.index"
METADATA_FILE = "documents_cache.jsonl" 

# Models for Embedding and Reranking (Crucial for stability)
EMBED_MODEL = "all-MiniLM-L6-v2" 
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     
LLM_MODEL = "google/flan-t5-base"        

# Chunking Parameters
PARENT_CHUNK_SEPARATOR = r'\n\s*\n' # Splits by double newline (paragraph/section breaks)
SMALL_CHUNK_SIZE = 700 
CHUNK_OVERLAP_SENTENCES = 2 
MIN_CHUNK_LEN = 40 

# Retrieval and Reranking Hyperparameters
TOP_K_INITIAL = 15 # Initial retrieval size (high recall)
TOP_K_FINAL = 3    # Final number of *Parent Contexts* sent to the LLM (high precision)
SIMILARITY_THRESHOLD = 0.20 
BATCH_SIZE = 64
# -------------------------------------


# --- Global Model Initialization ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models outside of FastAPI request handling for efficiency
try:
    embedder: SentenceTransformer = SentenceTransformer(EMBED_MODEL)
    reranker: CrossEncoder = CrossEncoder(RERANKER_MODEL, device=device) 
except Exception as e:
    print(f"Error loading models. Ensure dependencies are installed and check internet connection: {e}")
    exit()

qa_model = None  
index: Optional[faiss.Index] = None
metadata: List[Dict] = [] 
parent_cache: Dict[str, str] = {} # Cache for Parent Documents

app = FastAPI(title="DBS Customer Query Responder (Advanced RAG)")

class Query(BaseModel):
    question: str

# --- Utility Functions ---

def clean_text(t: str) -> str:
    if not t:
        return ""
    # Normalize whitespace, preserving structural double newlines
    t = t.replace("\r", "\n").replace("\n\n\n", "\n\n").strip() 
    t = re.sub(r"[ \t]+", " ", t) 
    return t

def chunk_text_sentences(text: str, size_chars: int, overlap_sentences: int) -> List[str]:
    """
    Generates sentence-aware Small Chunks for indexing.
    """
    if not text: return []
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback if punkt is not downloaded (should be fixed in build_index)
        nltk.download("punkt", quiet=True)
        sentences = sent_tokenize(text)
        
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        s = sent.strip()
        if not s: continue
        
        if current_len + len(s) + len(current) <= size_chars or not current:
            current.append(s)
            current_len += len(s)
        else:
            chunks.append(" ".join(current))
            
            # Apply sentence overlap
            overlap_sents = current[-overlap_sentences:] if len(current) >= overlap_sentences else current[:]
            current = overlap_sents + [s]
            current_len = sum(len(x) for x in current)

    if current:
        chunks.append(" ".join(current))

    final = [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]
    return final

# --- Indexing Functions ---

def build_index():
    """
    Builds the FAISS index using Structural Chunking (Parent Document concept).
    """
    print("Downloading NLTK punkt tokenizer (if needed)...")
    nltk.download("punkt", quiet=True)

    docs: List[Dict] = []
    global parent_cache 
    parent_cache.clear() 

    file_paths = glob.glob(os.path.join(DOC_FOLDER, "*.txt")) 
    
    if not file_paths:
        raise ValueError(f"No .txt files found in '{DOC_FOLDER}'.")

    for file_path in file_paths:
        try:
            # 1. Extract and Clean Raw Content from TXT
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            
            raw_content = clean_text(raw_content)

            # 2. Structural Split into Parent Chunks
            parent_chunks = re.split(PARENT_CHUNK_SEPARATOR, raw_content) 
            
            for p_chunk in parent_chunks:
                p_chunk = p_chunk.strip()
                if not p_chunk: continue

                parent_id = str(uuid.uuid4())
                parent_cache[parent_id] = p_chunk # Store Parent Content in memory

                # 3. Split Parent into Small Chunks (for embedding)
                small_chunks = chunk_text_sentences(p_chunk, SMALL_CHUNK_SIZE, CHUNK_OVERLAP_SENTENCES)
                
                # 4. Store metadata for Small Chunks, linking to Parent
                for s_chunk in small_chunks:
                    docs.append({
                        "source": os.path.basename(file_path), 
                        "content": s_chunk, 
                        "parent_id": parent_id 
                    })

        except Exception as e:
            print(f"Warning: failed to process {file_path}: {e}")

    if not docs:
        raise ValueError(f"No extractable text chunks present after structural splitting.")

    texts = [d["content"] for d in docs]
    print(f"Embedding {len(texts)} small chunks with '{EMBED_MODEL}' (batch_size={BATCH_SIZE})...")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=BATCH_SIZE)

    # FAISS Index Creation and Saving
    faiss.normalize_L2(embeddings.astype(np.float32))
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings.astype(np.float32))
    faiss.write_index(idx, INDEX_FILE)
    print(f"Saved index to '{INDEX_FILE}'")

    # Save the small chunk metadata file
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote metadata (small chunks) to '{METADATA_FILE}'")
    
    print(f"Indexed {len(texts)} small chunks successfully.")
    print(f"Created {len(parent_cache)} Parent Contexts.")


def load_index_and_metadata() -> Tuple[faiss.Index, List[Dict], Dict[str, str]]:
    """Load FAISS index and metadata for runtime."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("Index or metadata not found. Run with --build first.")
    
    idx = faiss.read_index(INDEX_FILE)
    docs = []
    temp_parent_cache = {}
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            docs.append(doc)
            
            parent_id = doc.get("parent_id")
            content = doc.get("content", "")
            if parent_id and content:
                # Aggregate small chunks by parent_id to reconstruct the Parent Context
                temp_parent_cache.setdefault(parent_id, set()).add(content)

    # Finalize parent_cache: join the small chunks back into a large context
    final_parent_cache = {
        pid: " ".join(sorted(list(chunks))) # Sort chunks to ensure consistent order
        for pid, chunks in temp_parent_cache.items()
    }
    
    return idx, docs, final_parent_cache

# --- RAG Endpoint ---

@app.post("/ask")
def ask(query: Query):
    global index, metadata, qa_model, parent_cache
    if index is None or metadata is None or qa_model is None or not parent_cache:
        raise HTTPException(status_code=500, detail="Server not fully initialized. Start server after building index.")

    q_text = clean_text(query.question)

    # 1. Initial Retrieval (Bi-Encoder Search on Small Chunks)
    q_emb = embedder.encode([q_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb.astype(np.float32))

    D, I = index.search(q_emb.astype(np.float32), k=TOP_K_INITIAL)  
    scores = D[0].tolist()
    indices = I[0].tolist()

    retrieval_candidates = []
    for sc, idx in zip(scores, indices):
        if idx < 0 or idx >= len(metadata): continue
        if sc < SIMILARITY_THRESHOLD: continue
        
        candidate = metadata[idx] 
        retrieval_candidates.append({
            "query": q_text,
            "content": candidate["content"], # Small Chunk
            "source": candidate["source"],
            "parent_id": candidate["parent_id"],
            "bi_encoder_score": float(sc) 
        })

    if not retrieval_candidates:
        return {"answer": "I don’t know based on the provided documents.", "sources": []}

    # 2. Reranking (Cross-Encoder Re-evaluation)
    rerank_input = [[q_text, r['content']] for r in retrieval_candidates]
    rerank_scores = reranker.predict(rerank_input, show_progress_bar=False)
    
    for i, score in enumerate(rerank_scores):
        retrieval_candidates[i]['rerank_score'] = float(score)

    reranked_candidates = sorted(retrieval_candidates, key=lambda x: x['rerank_score'], reverse=True)

    # 3. Parent Document Augmentation & Final Selection (The "Fix")
    final_parent_contexts = {}
    
    for candidate in reranked_candidates:
        parent_id = candidate['parent_id']
        
        # Stop if we hit our limit
        if len(final_parent_contexts) >= TOP_K_FINAL:
            break

        # If we already processed this Parent, skip
        if parent_id in final_parent_contexts:
            continue
            
        # Retrieve the large, coherent parent context
        parent_content = parent_cache.get(parent_id, "Context not found.")
        
        final_parent_contexts[parent_id] = {
            "source": candidate['source'],
            "content": parent_content, # Pass the large context to the LLM
            "score": candidate['rerank_score'] 
        }
        
    retrieved_contexts = list(final_parent_contexts.values())

    # 4. Generation (LLM Synthesis with Strong Guardrail)
    context_parts = []
    for r in retrieved_contexts:
        # Use the large parent content for context
        context_parts.append(f"[Source: {r['source']}] {r['content']}")

    context = "\n\n".join(context_parts)

    prompt = (
        "You are DBS Customer Support. Your goal is to provide precise, actionable, and safe banking information. "
        "Answer the question using ONLY the information provided in the Context section below. "
        "Do NOT introduce any external knowledge. "
        "If the context does not contain the answer, or if the information is incomplete, you MUST reply 'I cannot confirm that information based on the provided documents and recommend checking the official DBS website or contacting customer support.' "
        "Be concise and actionable.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {q_text}\nAnswer:"
    )

    out = qa_model(prompt, max_new_tokens=200, do_sample=False)
 
    generated_text = None
    if isinstance(out, list) and len(out) > 0:
        generated_text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
    else:
        generated_text = str(out)

    answer = generated_text.strip()
    
    return {"answer": answer, "sources": retrieved_contexts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build FAISS index (reads files and creates cache)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.build:
        print("Building index from files in:", DOC_FOLDER)
        build_index()
        print("Done. Now start the server with: python your_script_name.py")
    else:
        
        print("Loading index, metadata, and parent cache...")
        index, metadata, parent_cache = load_index_and_metadata()
        print(f"Loaded index with {len(metadata)} small chunks and {len(parent_cache)} Parent Contexts.")

        
        print(f"Initializing LLM pipeline: {LLM_MODEL} (this may take a while)...")
        # Initialize LLM after loading the index
        qa_model = pipeline("text2text-generation", model=LLM_MODEL, device=0 if device == "cuda" else -1)
        print("LLM ready. Starting FastAPI server...")

        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)