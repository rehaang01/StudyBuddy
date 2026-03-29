"""
upload.py
POST /upload/file  — Full RAG ingestion pipeline:
  1. Upload file to Azure Blob Storage
  2. Extract text per page via Azure Document Intelligence (extract_pages_from_url)
  3. Chunk by paragraph + page boundaries (chunk_by_paragraphs)
  4. Embed each chunk with Gemini gemini-embedding-001 — PARALLEL via asyncio.gather
  5. Store embeddings in Azure AI Search (scoped to conversation_id, with page_number)
"""

import asyncio

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import RedirectResponse
from app.models import UploadResponse
from app.services.blob_service import upload_file_to_blob, generate_fresh_sas_url
from app.services.doc_intelligence_service import extract_pages_from_url
from app.services.ai_service import embed_text
from app.services.search_service import store_chunks, create_index_if_not_exists
from app.utils.chunking import chunk_by_paragraphs
from app.services.cosmos_service import ensure_conversation, save_message

router = APIRouter(prefix="/upload", tags=["Upload"])

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "webp", "tiff"}
MAX_FILE_SIZE_MB = 20


@router.get("/view-file")
async def view_file(blob_name: str = Query(..., description="Permanent blob path stored at upload time")):
    """
    Permanent file-access proxy.

    Generates a fresh 1-hour SAS URL on-demand and redirects the browser to it.
    Because this endpoint URL never changes (only the SAS it generates does),
    files stored in Cosmos DB remain openable forever — no matter how old they are.

    The browser follows the 302 redirect transparently, so the user just sees
    their file open normally.
    """
    try:
        fresh_sas_url = generate_fresh_sas_url(blob_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not generate file URL: {str(e)}")
    return RedirectResponse(url=fresh_sas_url, status_code=302)


@router.post("/blob-only")
async def upload_blob_only(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    """
    Lightweight upload — stores file in Azure Blob only.
    Returns blob_url immediately. No text extraction or indexing.
    The full RAG pipeline runs later inside /chat/message when the user sends their query.
    """
    filename = file.filename or "upload"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '.{extension}' not supported.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File size {size_mb:.1f} MB exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # ── Page limit check for PDFs ─────────────────────────────────────────────
    if extension == "pdf":
        try:
            import fitz
            pdf_doc    = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = len(pdf_doc)
            pdf_doc.close()
            if page_count > 35:
                raise HTTPException(
                    status_code=422,
                    detail=f"PAGE_LIMIT_EXCEEDED:{page_count}",
                )
        except HTTPException:
            raise   # re-raise the 422 — don't swallow it
        except Exception as e:
            print(f"[upload] Could not count PDF pages: {e}")
            # non-fatal — let the upload proceed if page counting fails

    try:
        blob_info = upload_file_to_blob(file_bytes, filename, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blob upload failed: {str(e)}")

    return {
        "blob_url": blob_info["blob_url"],    # short-lived SAS — frontend uses for immediate RAG
        "blob_name": blob_info["blob_name"],  # permanent identifier — frontend uses for proxy URL
        "filename": filename,
    }


@router.post("/file", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    conversation_id: str = Form(default="no-conversation"),
):
    """
    Upload a study document and run the full RAG pipeline.

    - Accepts: PDF, PNG, JPG, JPEG, WEBP, TIFF (max 20 MB)
    - conversation_id scopes the stored chunks to this chat session only,
      so files from other chats are never mixed in during retrieval.
    - Returns: file_id, blob_url, number of chunks indexed
    """

    # ── Validate file type ────────────────────────────────────────────────────
    filename = file.filename or "upload"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{extension}' is not supported. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Read file bytes ───────────────────────────────────────────────────────
    file_bytes = await file.read()

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size {size_mb:.1f} MB exceeds the {MAX_FILE_SIZE_MB} MB limit.",
        )

    # ── Step 1: Upload to Azure Blob Storage ─────────────────────────────────
    try:
        blob_info = upload_file_to_blob(file_bytes, filename, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blob upload failed: {str(e)}")

    blob_url = blob_info["blob_url"]
    file_id  = blob_info["file_id"]

    # ── Step 2: Extract text per page via Document Intelligence ──────────────
    # run_in_executor prevents blocking the async event loop during OCR
    try:
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, lambda: extract_pages_from_url(blob_url))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the uploaded file. "
                   "Please check the file is readable.",
        )

    # ── Step 3: Chunk by paragraphs (page-boundary aware) ────────────────────
    chunk_dicts  = chunk_by_paragraphs(pages)
    chunks       = [c["text"] for c in chunk_dicts]
    page_numbers = [c["page_number"] for c in chunk_dicts]

    # ── Step 4: Embed each chunk with Gemini — PARALLEL ──────────────────────
    # asyncio.gather() runs all embed calls concurrently on a thread pool.
    # Semaphore(5) caps concurrent calls to avoid Gemini rate limits.
    try:
        semaphore = asyncio.Semaphore(10)
        async def embed_one(c):
            async with semaphore:
                return await asyncio.get_event_loop().run_in_executor(None, lambda: embed_text(c))
        embeddings = await asyncio.gather(*[embed_one(c) for c in chunks])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # ── Step 5: Store in Azure AI Search ─────────────────────────────────────
    try:
        create_index_if_not_exists()
        store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            user_id=user_id,
            conversation_id=conversation_id,
            file_id=file_id,
            filename=filename,
            page_numbers=page_numbers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search indexing failed: {str(e)}")

    # ── Persist upload confirmation to Cosmos so it survives page refresh ────
    if conversation_id and conversation_id != "no-conversation":
        try:
            await ensure_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                title="",
            )
            await save_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=(
                    f"📎 I've processed **{filename}** ({len(chunks)} chunks indexed). "
                    f"You can now ask me questions about it, or generate a flowchart / diagram from it!"
                ),
            )
        except Exception:
            pass  # Non-critical — don't fail the upload over a Cosmos write

    return UploadResponse(
        file_id=file_id,
        filename=filename,
        blob_url=blob_url,
        chunks_stored=len(chunks),
        message=f"Successfully processed '{filename}' — {len(chunks)} chunks indexed.",
    )
