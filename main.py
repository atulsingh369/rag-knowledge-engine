from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pdfplumber
import os
import uuid
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from pinecone import Pinecone
import google.generativeai as genai

# ✅ Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = "primary-knowledge-base"
if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set in .env file")

pc = Pinecone(api_key=pinecone_api_key)
print(pc.describe_index("primary-knowledge-base"))

# ✅ Gemini setup
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY must be set in .env file")

genai.configure(api_key=gemini_api_key)
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768

# ✅ Directory for uploads
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ✅ Chunk text
def chunk_text(text, chunk_size=800):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ✅ Generate Gemini embeddings
def get_embeddings(texts):
    embeddings = []
    for t in texts:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=t,
            output_dimensionality=EMBEDDING_DIM
        )
        embeddings.append(response["embedding"])
    return embeddings

# ✅ Root endpoint
@app.get("/")
def root():
    return {"message": "Hello from FastAPI on Render!"}

# ✅ Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# ✅ Upload PDF endpoint
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "Uploaded file must have a filename."})
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    text = extract_text_from_pdf(file_location)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    index = pc.Index(pinecone_index_name)
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": f"{file.filename}_{uuid.uuid4()}",
            "values": embedding,
            "metadata": {
                "source": file.filename,
                "text": chunk,
                "type": "pdf",
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        })

    index.upsert(vectors=vectors)
    return JSONResponse(content={"message": f"✅ Stored {len(chunks)} file chunks in Pinecone"})

# ✅ Add text endpoint
@app.post("/add-text")
async def add_text(text: str = Form(...)):
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    index = pc.Index(pinecone_index_name)
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": f"text_{uuid.uuid4()}",
            "values": embedding,
            "metadata": {
                "source": "manual",
                "text": chunk,
                "type": "manual",
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        })

    index.upsert(vectors=vectors)
    return JSONResponse(content={"message": f"✅ Stored {len(chunks)} text chunks in Pinecone"})

# ✅ Query RAG endpoint
@app.post("/query")
async def query_rag(
    query: str = Form(...),
    type_filter: Optional[str] = Form(None),
    source_filter: Optional[str] = Form(None)
):
    try:
        print("▶️ Incoming query:", query)
        print("▶️ Type Filter:", type_filter, "Source Filter:", source_filter)

        # Gemini query embedding
        query_embedding = get_embeddings([query])[0]

        indexes = pc.list_indexes().names()
        if pinecone_index_name not in indexes:
            return JSONResponse(
                status_code=404,
                content={"error": f"Index '{pinecone_index_name}' not found", "available_indexes": indexes}
            )

        index = pc.Index(pinecone_index_name)

        # Metadata filter
        metadata_filter = {}
        if type_filter:
            metadata_filter["type"] = type_filter
        if source_filter:
            metadata_filter["source"] = source_filter

        print("▶️ Metadata filter:", metadata_filter)

        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=metadata_filter if metadata_filter else None
        )

        return JSONResponse(content=jsonable_encoder(results))

    except Exception as e:
        print("❌ Error in /query:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
