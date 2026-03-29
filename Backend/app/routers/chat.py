import asyncio
import io
import json
import re
import uuid

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from app.services.doc_intelligence_service import extract_text_from_url, extract_pages_from_url
from app.utils.chunking import chunk_text, chunk_by_paragraphs
from app.models import ChatRequest, ChatHistoryResponse, ChatMessage
from app.services.ai_service import (
    embed_query, embed_text, chat_stream, infer_topic_from_messages,
    classify_intent,
    generate_quiz_questions,
    generate_mermaid,
    generate_image as generate_image_bytes,
    extract_document_context,
    classify_search_intent,
    get_provider,
)
from app.services.search_service import (
    retrieve_chunks, retrieve_chunks_hybrid, store_chunks,
    create_index_if_not_exists, conversation_has_documents,
    retrieve_chunks_smart,get_conversation_filenames,
    retrieve_all_chunks_ordered,
)
from app.services.cosmos_service import (
    create_conversation,
    save_message,
    get_messages,
    get_conversation_full,
    list_conversations,
    update_message_json,
    update_message_content,
    save_quiz,
    save_diagram,
    save_image_diagram,
    rename_conversation,
    delete_conversation,
    star_conversation,
    set_conversation_provider,
)
from app.services.blob_service import upload_generated_image_to_blob
from app.services.study_plan_service import create_study_plan
from app.services.translator_service import translate_text
from app.services.tts_service import synthesize_speech
from app.services.web_search_service import web_search, build_search_context, image_search, youtube_search_api, youtube_search
from app.utils.document_resolver import resolve_document_filter
from app.services.settings_service import get_settings
from app.utils.curriculum_profiles import get_curriculum_context
from app.utils.message_content import extract_message_text_content, parse_structured_message

router = APIRouter(prefix="/chat", tags=["Chat"])

# ── Safety constants ──────────────────────────────────────────────────────────
# Matches REFUSAL_SENTINEL in both service files. chat.py checks generation
# function return values against this string to detect a model refusal.
_REFUSAL_SENTINEL = "__REFUSED__"

# The polite message shown to the user in the chat bubble when content is refused.
_REFUSAL_MESSAGE = (
    "I'm StudyBuddy, an educational assistant. "
    "I can't help with that topic. Please ask me something related to your studies!"
)

# Extra guard for chip-driven web search path (where classify_intent is skipped).
_WEB_BLOCKED_PATTERN = re.compile(
    r"\b("
    r"porn|pornography|adult\s*content|adult\s*films?|onlyfans|xxx|nsfw|erotic|nude|"
    r"sex\s*videos?|sexual\s*videos?|lustful|explicit\s*content"
    r")\b",
    re.IGNORECASE,
)


async def _yield_refusal(conversation_id: str, user_id: str):
    """
    Async generator that yields a polite refusal as a plain text SSE event,
    saves it to Cosmos, and closes the stream.
    Used by all dispatch functions and the classify_intent is_harmful check.
    """
    await save_message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=_REFUSAL_MESSAGE,
    )
    yield f"data: {json.dumps({'type': 'text', 'content': _REFUSAL_MESSAGE})}\n\n"
    yield "data: [DONE]\n\n"


def _is_blocked_web_query(query: str) -> bool:
    """Fast keyword guard for unsafe web-search requests in chip path."""
    return bool(_WEB_BLOCKED_PATTERN.search(query or ""))


# ── Auto-title generation (fire-and-forget) ──────────────────────────────────

def _generate_title_text(user_message: str, intent: str = "", filename: str = "") -> str:
    """
    Small synchronous LLM call — generates a concise chat title.
    Runs on a thread executor, never in the main async loop.
    Always uses Azure regardless of user's model choice (tiny call, ~100ms).
    """
    from app.services.azure_openai_service import _get_client, _chat_deployment

    context_parts = []
    if user_message.strip():
        context_parts.append(f"User message: {user_message[:200]}")
    if intent and intent != "chat":
        context_parts.append(f"Action requested: {intent.replace('_', ' ')}")
    if filename:
        context_parts.append(f"Uploaded file: {filename}")

    context = "\n".join(context_parts) or "General conversation"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_chat_deployment(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-6 word title for this student's chat conversation. "
                        "Respond with ONLY the title. No quotes, no punctuation at the end, no explanation. "
                        "Examples: 'Photosynthesis Quiz', 'Newton Laws Summary', 'EC342 Flowchart', "
                        "'Microprocessor Study Plan', 'Attention Paper Notes'"
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'").rstrip(".")
        return title[:50] if title else ""
    except Exception as e:
        print(f"[TITLE] Generation failed: {e}")
        return ""


async def _auto_title_conversation(
    conversation_id: str, user_id: str,
    user_message: str, intent: str = "", filename: str = "",
):
    """
    Fire-and-forget: generates a chat title via LLM and saves to Cosmos.
    If generation fails, the placeholder '…' remains and the frontend
    falls back to showing the first message text on next refresh.
    """
    try:
        loop = asyncio.get_event_loop()
        title = await loop.run_in_executor(
            None, lambda: _generate_title_text(user_message, intent, filename),
        )
        if title:
            await rename_conversation(conversation_id, user_id, title)
            print(f"[TITLE] Set title for {conversation_id}: {title!r}")
        else:
            # Clear placeholder so fallback logic kicks in
            await rename_conversation(conversation_id, user_id, "")
    except Exception as e:
        print(f"[TITLE] Failed to set title for {conversation_id}: {e}")


def _is_refusal_text(text: str) -> bool:
    """Detect refusal-like replies produced by upstream safety handlers/models."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.startswith(_REFUSAL_SENTINEL.lower()):
        return True
    if t == _REFUSAL_MESSAGE.lower():
        return True
    return (
        "i'm studybuddy, an educational assistant" in t
        and "can't help with that topic" in t
    )


def _build_web_search_answer_payload(
    *,
    query: str,
    answer: str,
    sources: list,
    images: list,
    videos: list,
    previous_raw: Optional[str] = None,
) -> str:
    """
    Builds the persisted JSON payload for web search answers.

    When `previous_raw` is provided (regeneration path), keep full regeneration
    history under `regen_versions` so response switching survives page reloads.
    """
    payload = {
        "__type": "web_search_answer",
        "query": query,
        "answer": answer,
        "sources": sources,
        "images": images,
        "videos": videos,
    }

    if not previous_raw:
        return json.dumps(payload)

    try:
        prev = json.loads(previous_raw) if isinstance(previous_raw, str) else {}
    except Exception:
        prev = {}

    versions = prev.get("regen_versions") if isinstance(prev.get("regen_versions"), list) else None
    if not versions:
        prev_answer = prev.get("answer") if isinstance(prev, dict) else ""
        versions = []
        if prev_answer:
            versions.append(
                {
                    "content": prev_answer,
                    "sources": prev.get("sources") or [],
                    "images": prev.get("images") or [],
                    "videos": prev.get("videos") or [],
                }
            )

    versions.append(
        {
            "content": answer,
            "sources": sources,
            "images": images,
            "videos": videos,
        }
    )

    payload["regen_versions"] = versions
    payload["active_regen_version_idx"] = max(0, len(versions) - 1)
    return json.dumps(payload)


def _build_regen_text_answer_payload(*, answer: str, previous_raw: Optional[str] = None) -> str:
    """
    Builds the persisted JSON payload for regenerated plain-text answers.

    The current answer stays at top-level `content` for fast reads while the
    full version history lives in `regen_versions` so reload/offline restore can
    rebuild the response switcher without extra backend calls.
    """
    payload = {
        "__type": "regen_text_answer",
        "content": answer,
    }

    if not previous_raw:
        return json.dumps(payload)

    parsed_prev = parse_structured_message(previous_raw)
    versions = (
        parsed_prev.get("regen_versions")
        if isinstance(parsed_prev, dict) and isinstance(parsed_prev.get("regen_versions"), list)
        else None
    )

    if not versions:
        previous_text = (
            extract_message_text_content(previous_raw)
            or (
                str(parsed_prev.get("content", "") or "").strip()
                if isinstance(parsed_prev, dict)
                else ""
            )
        )
        versions = [{"content": previous_text}] if previous_text else []

    versions.append({"content": answer})
    payload["regen_versions"] = versions
    payload["active_regen_version_idx"] = max(0, len(versions) - 1)
    return json.dumps(payload)


# ── Request models ────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    target_language: str


class TTSRequest(BaseModel):
    text: str
    language: str
    voice_style: str = "buttery"


class InferTopicRequest(BaseModel):
    messages: List[dict]


class RegenerateRequest(BaseModel):
    user_id: str
    conversation_id: str
    message_id: str  # ID of the assistant message to regenerate


# ── Helper: tag chunks with page numbers so Gemini knows which page ───────────

def _tag_chunks_with_pages(chunk_tuples: list) -> list:
    """
    Takes list of (text, page_number) tuples from retrieve_chunks_smart().
    Sorts by page number so Gemini receives content in document order,
    then prefixes each chunk with [Page N] so Gemini knows which page it came from.

    Returns plain list of tagged strings ready to pass to chat_stream().
    """
    if not chunk_tuples:
        return []
    sorted_tuples = sorted(
        [item for item in chunk_tuples if isinstance(item, tuple)],
        key=lambda x: (x[2] if len(x) > 2 else "", x[1])
    )
    result = []
    for item in sorted_tuples:
        text, page_num = item[0], item[1]
        filename = item[2] if len(item) > 2 else ""
        prefix = f"[File: {filename} | Page {page_num}]" if filename else f"[Page {page_num}]"
        result.append(f"{prefix}\n{text}")
    return result


# ── POST /chat/regenerate ─────────────────────────────────────────────────────

@router.post("/regenerate")
async def regenerate_message_endpoint(request: RegenerateRequest):
    """
    Regenerates a specific assistant message without creating new Cosmos records.

    Unlike POST /chat/message this endpoint:
      - Does NOT append a new user message to Cosmos (original is already there).
      - REPLACES the content of the existing assistant message (no duplicate).
      - Injects an explicit variation instruction so Gemini produces a different
        answer instead of repeating the previous one verbatim.
    """
    conv_data = await get_conversation_full(request.conversation_id)
    all_messages = conv_data["messages"]

    target_idx = next(
        (i for i, m in enumerate(all_messages) if m.get("id") == request.message_id),
        None,
    )
    if target_idx is None:
        raise HTTPException(status_code=404, detail="Message not found in conversation.")

    prior_messages = all_messages[:target_idx]
    preceding_user = next(
        (m for m in reversed(prior_messages) if m.get("role") == "user"), None
    )
    if not preceding_user:
        raise HTTPException(status_code=400, detail="No preceding user message found.")

    user_text = extract_message_text_content(preceding_user.get("content", ""))

    prev_response = all_messages[target_idx].get("content", "")

    # ── Check if the message being regenerated was a web search answer ─────────
    # If so, re-dispatch to _dispatch_web_search so we get fresh results,
    # fresh images, and fresh source citations instead of a plain text reply.
    is_web_search_regen = prev_response.strip().startswith('{"__type": "web_search_answer"')

    if is_web_search_regen:
        async def regen_stream():
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': request.conversation_id})}\n\n"
            async for evt in _dispatch_web_search(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                query=user_text,
                prior_messages=prior_messages,
                has_attachment=False,
                update_message_id=request.message_id,
                previous_message_content=prev_response,
            ):
                yield evt

        return StreamingResponse(
            regen_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    q_text = user_text or "key concepts"
    context_chunks: list = []
    try:
        loop = asyncio.get_event_loop()
        has_docs = await loop.run_in_executor(
            None,
            lambda: conversation_has_documents(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            ),
        )
        if has_docs:
            # Detect if original query wanted full document coverage
            regen_doc_context = {}
            try:
                regen_doc_context = await loop.run_in_executor(
                    None, lambda: extract_document_context(user_text)
                )
            except Exception:
                pass

            regen_scope = regen_doc_context.get("scope") or "topic"

            if regen_scope == "document":
                # Full coverage — fetch ALL chunks ordered by file+page
                raw_chunks = retrieve_all_chunks_ordered(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                )
            else:
                query_embedding = await loop.run_in_executor(
                    None, lambda: embed_query(q_text)
                )
                raw_chunks = retrieve_chunks_smart(
                    query_embedding=query_embedding,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    top_k=10,
                )

            # Tag with file+page labels so AI knows which file/page each chunk is from
            context_chunks = _tag_chunks_with_pages(raw_chunks)
    except Exception:
        pass  # RAG failure is non-fatal — fall back to general knowledge

    async def regen_stream():
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': request.conversation_id})}\n\n"

        if prev_response.strip():
            regen_question = (
                f"{user_text}\n\n"
                f"[Regeneration instruction: Your previous answer covered this topic already. "
                f"Please respond with a fresh explanation — use different phrasing, "
                f"different examples or analogies, and a new angle or structure. "
                f"Do NOT repeat the same wording as before.]"
            )
        else:
            regen_question = user_text

        full_reply = ""
        try:
            for chunk in chat_stream(
                question=regen_question,
                context_chunks=context_chunks,
                history=prior_messages,
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Mid-stream refusal fallback
        if _REFUSAL_MESSAGE in full_reply and full_reply.strip() != _REFUSAL_MESSAGE:
            full_reply = _REFUSAL_MESSAGE

        new_content = _build_regen_text_answer_payload(
            answer=full_reply,
            previous_raw=prev_response,
        )

        await update_message_content(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            message_id=request.message_id,
            new_content=new_content,
        )
        yield f"data: {json.dumps({'type': 'message_saved', 'message_id': request.message_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        regen_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── POST /chat/message ────────────────────────────────────────────────────────

@router.post("/message")
async def chat_message(request: ChatRequest):
    """
    Unified chat endpoint with intent classification and dispatch.

    1. Fetches history + pending_intent in ONE Cosmos read.
    2a. [OPT 1] If intent_hint set (chip click) → skip Gemini classify call entirely.
    2b. [OPT 2] Otherwise → run classify_intent + embed_query IN PARALLEL on thread pool.
    3. Clarification → streams question + stores pending_intent in Cosmos.
    4. [OPT 3] Smart retrieval: page filter, hybrid search, dynamic top_k, skip when scope=general.
    5. Feature intent → dispatches to quiz / diagram / image / study_plan.
    6. Default → existing RAG chat stream.
    """

    # ── Ensure conversation ───────────────────────────────────────────────────
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await create_conversation(request.user_id, title="…")

    # ── History + Settings in parallel (two independent Cosmos containers) ────
    # get_conversation_full → conversations container (keyed by conversation_id)
    # get_settings           → settings container (keyed by user_id)
    # Zero data dependency — safe to run simultaneously. Saves ~50-80ms.
    async def _fetch_settings_safe():
        try:
            return await get_settings(request.user_id)
        except Exception:
            return None

    conv_data, _user_settings = await asyncio.gather(
        get_conversation_full(conversation_id),
        _fetch_settings_safe(),
    )

    prior_messages = conv_data["messages"]
    pending_intent = conv_data["pending_intent"]

    # ── Curriculum context (pure dict lookup — ~0.01ms, no I/O) ──────────────
    try:
        _curr_context: str | None = get_curriculum_context(
            board=(_user_settings or {}).get("curriculum_board"),
            grade=(_user_settings or {}).get("curriculum_grade"),
            enabled=bool((_user_settings or {}).get("curriculum_enabled", False)),
        ) if _user_settings else None
    except Exception:
        _curr_context = None
    print(f"[CURRICULUM] user={request.user_id} | context={'SET (' + _curr_context[:60] + '...)' if _curr_context else 'None'}")

    # ── Resolve request-scoped AI provider ────────────────────────────────────
    # get_provider() returns the correct module for this request only.
    # Falls back to the server-wide default (Azure) when model_provider is None.
    # Embeddings always use Azure regardless — get_provider() does not affect
    # embed_query / embed_text which are imported directly from ai_service.
    _provider = get_provider(request.model_provider)
    _provider_name = "GEMINI" if "gemini" in getattr(_provider, "__name__", "") else "AZURE"
    print(f"[PROVIDER] user={request.user_id} | conv={conversation_id} | model={_provider_name} (requested={request.model_provider!r})")

    # ── Persist sticky provider choice to the conversation document ───────────
    # Fire-and-forget: we don't await so it never adds latency to the stream.
    # Only writes when the value actually changed (set_conversation_provider is
    # a no-op when the stored value already matches).
    if request.model_provider and conversation_id:
        asyncio.ensure_future(
            set_conversation_provider(
                conversation_id=conversation_id,
                user_id=request.user_id,
                model_provider=request.model_provider,
            )
        )

    pre_embedding = None  # may be pre-computed by Opt 2 below
    

    if request.intent_hint:
        # Intent known from chip — build classification dict with no Gemini call
        msg_lower = (request.message or "").lower()
        num_q_match = re.search(r"(\d+)\s*(questions?|ques|qs\b|q\b|mcqs?)", msg_lower)
        # Parse weeks for study_plan chip: "4 weeks", "2 months" → weeks. Default 4.
        weeks_match = re.search(r'(\d+)\s*(week|month)', msg_lower)
        weeks_val = 4
        if weeks_match:
            n = int(weeks_match.group(1))
            weeks_val = n * 4 if "month" in weeks_match.group(2) else n
        # Parse timer: "30 seconds", "1 minute", "2 mins" → seconds
        timer_match = re.search(r'(\d+)\s*(second|sec|minute|min)', msg_lower)
        timer_val = None
        if timer_match:
            n = int(timer_match.group(1))
            timer_val = n * 60 if "min" in timer_match.group(2) else n

        # ── Override fields take priority (used by retake to avoid encoding
        #    count/timer in the message text, which polluted quiz titles) ──────
        if request.num_questions_override is not None:
            num_q_match = None  # suppress regex result
            num_questions_val = request.num_questions_override
        else:
            num_questions_val = int(num_q_match.group(1)) if num_q_match else 5

        if request.timer_seconds_override is not None:
            timer_val = request.timer_seconds_override

        # Topic is missing only when: no message text AND no file attached.
        # If a file is attached, topic will be inferred from the document.
        # web_search is special: the message itself is the query — never ask for clarification.
        has_attachment = bool(request.filename or request.attachments)
        no_topic = (
            not request.message.strip()
            and not has_attachment
            and request.intent_hint != "web_search"
        )

        # ── Clean topic from raw message ───────────────────────────────────────
        # The chip already tells us the INTENT (quiz / flowchart / etc.), so we
        # must strip any intent-describing words from the user's message to get
        # a clean topic string.  Without this, a message like
        # "give me a flowchart on photosynthesis" with the Quiz chip selected
        # would pass the entire sentence as the topic to generate_quiz_questions,
        # confusing Gemini and producing irrelevant / garbled questions.
        #
        # Strategy: remove common intent-verb phrases and prepositions that
        # precede the actual topic, then trim whitespace. Apply TRAILING_STRIP
        # in a loop to handle multiple trailing quantity/time phrases.
        _INTENT_STRIP = re.compile(
            r'^(please\s+)?(can\s+you\s+)?'
            r'(give\s+me\s+(a\s+)?|make\s+(a\s+|me\s+(a\s+)?)?|'
            r'create\s+(a\s+)?|generate\s+(a\s+)?(fresh\s+)?|show\s+me\s+(a\s+)?|build\s+(a\s+)?|'
            r'i\s+want\s+(a\s+)?|produce\s+(a\s+)?)?'
            r'(timed\s+)?(quiz\s+me\s+on|quiz|test|questions?|mcq|flowchart|flow\s+chart|mindmap|mind\s+map|'
            r'diagram|image\s+of|image|picture\s+of|picture|illustration|study\s+plan|plan)\s*'
            r'(me\s+on\s+|about|on\s*:?\s*|for|of|regarding|related\s+to|covering)?\s*',
            re.IGNORECASE,
        )
        # Strip trailing quantity/time phrases — loop until stable so that
        # "rohit sharma, 3 questions, 60 seconds" strips both suffixes cleanly.
        _TRAILING_STRIP = re.compile(
            r'[\s,]*(with\s+)?\d+\s*(questions?|ques|qs\b|q\b|mcqs?|items?|mins?|minutes?|seconds?|secs?)\s*$',
            re.IGNORECASE,
        )
        raw_msg = request.message.strip()
        # Loop intent strip to handle legacy doubled prefixes (e.g. old retake messages
        # that inadvertently stacked "Generate a fresh quiz on:" twice).
        cleaned_topic = raw_msg
        while True:
            stripped = _INTENT_STRIP.sub("", cleaned_topic).strip()
            if stripped == cleaned_topic:
                break
            cleaned_topic = stripped
        # Loop trailing strip until nothing more is removed
        while True:
            stripped = _TRAILING_STRIP.sub("", cleaned_topic).strip().rstrip(",").strip()
            if stripped == cleaned_topic:
                break
            cleaned_topic = stripped
        # Fall back to full message only if stripping removed everything
        topic_val = cleaned_topic or raw_msg or ("[from_document]" if has_attachment else None)

        # ── Extract document/page context via small targeted LLM call ────────────
        # Only fires when chip is selected and user has typed a message.
        # Gives chip path the same page/document awareness as natural language path.
        page_numbers_chip = []
        # web_search never uses extract_document_context output — its dispatch
        # receives only query/user_id/conversation_id, not topic/page_numbers/scope.
        # Skipping saves ~150ms LLM call on every web search request.
        # For web_search: force scope="general" so pre-stream retrieval is skipped too.
        if request.intent_hint == "web_search":
            chip_scope = "general"
        else:
            chip_scope = "topic"
            if request.message.strip():
                try:
                    loop = asyncio.get_event_loop()
                    doc_context = await loop.run_in_executor(
                        None,
                        lambda: extract_document_context(request.message)
                    )
                    page_numbers_chip = doc_context.get("page_numbers") or []
                    doc_ref           = doc_context.get("document_reference") or ""
                    clean_topic       = doc_context.get("clean_topic") or ""
                    chip_scope        = doc_context.get("scope") or "topic"

                    # Use clean topic (strips "in document 1", "on page 3" etc.)
                    if clean_topic:
                        topic_val = clean_topic

                    # Merge doc reference into topic_val so resolve_document_filter
                    # can match it against actual filenames later in dispatch
                    if doc_ref and doc_ref.lower() not in (topic_val or "").lower():
                        topic_val = f"{topic_val} {doc_ref}".strip() if topic_val else doc_ref

                except Exception:
                    pass   # non-fatal — falls back to empty

        # ── Attachment override: file uploaded with chip → never skip retrieval ──
        # When a file is attached alongside ANY chip (quiz, flowchart, mindmap,
        # study_plan, image), the user clearly wants that file used.
        # extract_document_context may return scope="general" for short/vague
        # message text (e.g. "in 3 questions"), but the file attachment signals
        # intent to work with the document.
        if has_attachment and chip_scope == "general":
            chip_scope = "document"

        classification = {
            "intent":               request.intent_hint,
            "topic":                topic_val,
            "topic_source":         "message" if request.message.strip() else ("document" if has_attachment else "null"),
            "num_questions":        num_questions_val,
            "timeline_weeks":       weeks_val,
            "hours_per_week":       None,
            "timer_seconds":        timer_val,
            "needs_clarification":  no_topic,
            "clarification_question": (
                f"What topic would you like for your "
                f"{request.intent_hint.replace('_', ' ')}?"
                if no_topic else None
            ),
            # default values for new fields when chip path is used
            "page_numbers": page_numbers_chip, "keywords": [], "query_type": "broad",
            "top_k_hint": "medium", "scope": chip_scope,
            "response_format": "paragraph", "detail_level": "detailed",
            "language_style": "formal", "is_comparison": False,
            "entities": [], "needs_document": True,
            "is_followup": False, "refers_to_previous": False,
        }
    else:
        # Natural language path — run classify + embed concurrently on thread pool
        loop = asyncio.get_event_loop()
        classification, pre_embedding = await asyncio.gather(
            loop.run_in_executor(None, lambda: _provider.classify_intent(
                message=request.message,
                intent_hint=None,
                conversation_history=prior_messages,
                attached_filename=request.filename,
                pending_intent=pending_intent,
            )),
            loop.run_in_executor(None, lambda: embed_query(
                request.message or "key concepts"
            )),
        )

    # ── Safety gate (NLP path only) ───────────────────────────────────────────
    # The chip path skips classify_intent entirely, so is_harmful is only
    # available here. The chip path is protected downstream by SAFETY_BLOCK
    # sentinel checks in each dispatch function.
    if classification.get("is_harmful"):
        harm_reason = classification.get("harm_reason") or "inappropriate content"
        print(f"[SAFETY] Blocked harmful request. Reason: {harm_reason}. Message: {request.message[:80]}")
        await save_message(
            conversation_id=conversation_id,
            user_id=request.user_id,
            role="user",
            content=request.message,   # user_content not built yet at this point — use raw message
            pending_intent_update=None,
        )

        # Set a safe generic title so the sidebar never shows the blocked text
        if not prior_messages:
            await rename_conversation(conversation_id, request.user_id, "Conversation")

        async def safety_refusal_stream():
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"
            async for evt in _yield_refusal(conversation_id, request.user_id):
                yield evt

        return StreamingResponse(
            safety_refusal_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    intent              = classification["intent"]
    topic_raw           = classification.get("topic") or ""
    num_questions       = classification["num_questions"]
    timeline_weeks      = classification.get("timeline_weeks")
    hours_per_week      = classification.get("hours_per_week")
    timer_seconds       = classification.get("timer_seconds")
    needs_clarification = classification["needs_clarification"]
    clarification_q     = classification.get("clarification_question") or "Could you tell me more?"

    # new classification fields
    page_numbers        = classification.get("page_numbers") or []
    keywords            = classification.get("keywords") or []
    query_type          = classification.get("query_type") or "broad"
    top_k_hint          = classification.get("top_k_hint") or "medium"
    scope               = classification.get("scope") or "topic"
    response_format     = classification.get("response_format") or "paragraph"
    detail_level        = classification.get("detail_level") or "detailed"
    language_style      = classification.get("language_style") or "formal"
    is_comparison       = classification.get("is_comparison") or False
    entities            = classification.get("entities") or []
    needs_document      = classification.get("needs_document", True)
    is_followup         = classification.get("is_followup") or False

    print(f"[DEBUG] page_numbers={page_numbers}, scope={scope}, top_k_hint={top_k_hint}, query_type={query_type}, needs_document={needs_document}")
    print(f"[DEBUG] response_format={response_format}, detail_level={detail_level}, language_style={language_style}")

    # "[from_document]" is sentinel meaning "derive topic from uploaded material"
    topic = "" if topic_raw == "[from_document]" else topic_raw

    # ── Build user_content for Cosmos ─────────────────────────────────────────
    has_chip = bool(request.intent_hint)
    has_att  = bool(request.attachments)

    if has_chip and has_att:
        user_payload = {
            "__type": "user_with_intent_and_attachments",
            "text": request.message,
            "intent_hint": request.intent_hint,
            "attachments": [
                {"name": a.name, "blob_name": a.blob_name, "blob_url": a.blob_url, "file_type": a.file_type}
                for a in request.attachments
            ],
        }
        if request.retake_of_quiz_id:
            user_payload["retake_of_quiz_id"] = request.retake_of_quiz_id
        user_content = json.dumps(user_payload)
    elif has_chip:
        user_payload = {
            "__type": "user_with_intent",
            "text": request.message,
            "intent_hint": request.intent_hint,
        }
        if request.retake_of_quiz_id:
            user_payload["retake_of_quiz_id"] = request.retake_of_quiz_id
        user_content = json.dumps(user_payload)
    elif has_att:
        user_content = json.dumps({
            "__type": "user_with_attachments",
            "text": request.message,
            "attachments": [
                {"name": a.name, "blob_name": a.blob_name, "blob_url": a.blob_url, "file_type": a.file_type}
                for a in request.attachments
            ],
        })
    else:
        user_content = request.message

    # ── Clarification short-circuit ───────────────────────────────────────────
    if needs_clarification:
        await save_message(
            conversation_id=conversation_id,
            user_id=request.user_id,
            role="user",
            content=user_content,
            pending_intent_update=intent if intent != "chat" else None,
        )
        await save_message(
            conversation_id=conversation_id,
            user_id=request.user_id,
            role="assistant",
            content=clarification_q,
        )

        async def clarification_stream():
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': clarification_q})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            clarification_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Save user message and clear pending_intent ────────────────────────────
    # For web_search with no attachment: run save_message in parallel with
    # conversation_has_documents so their costs overlap completely.
    # save_message (~50ms Cosmos) hides behind has_docs (~80ms Azure Search).
    # For all other intents: save_message runs sequentially as before.
    _pre_has_docs: bool | None = None  # None = not yet checked
    if intent == "web_search" and not bool(request.blob_url or request.attachments):
        _ws_loop = asyncio.get_event_loop()
        async def _save_user_msg():
            await save_message(
                conversation_id=conversation_id,
                user_id=request.user_id,
                role="user",
                content=user_content,
                pending_intent_update=None,
            )
        async def _check_has_docs():
            return await _ws_loop.run_in_executor(
                None,
                lambda: conversation_has_documents(
                    user_id=request.user_id,
                    conversation_id=conversation_id,
                ),
            )
        _, _pre_has_docs = await asyncio.gather(_save_user_msg(), _check_has_docs())
    else:
        await save_message(
            conversation_id=conversation_id,
            user_id=request.user_id,
            role="user",
            content=user_content,
            pending_intent_update=None,
        )

    # ── Smart retrieval ───────────────────────────────────────────────────────
    # - scope=general / needs_document=False → skip Azure Search entirely
    # - page_numbers present → filter to specific pages only
    # - keywords present → use hybrid (BM25 + vector)
    # - scope=document → top_k=50 to fetch all chunks
    # NOTE: When file is uploaded inline with message, retrieval runs AGAIN
    # inside event_stream() AFTER chunks are stored. This pre-stream retrieval
    # only hits when docs were uploaded in a previous message.
    q_text = request.message or topic or "key concepts"
    query_embedding = None
    context_chunks  = []
    try:
        skip_retrieval = (scope == "general") or (not needs_document)
        print(f"[DEBUG-RETRIEVAL] page_numbers={page_numbers}, scope={scope}, skip={skip_retrieval}, use_hybrid={bool(keywords)}")

        if not skip_retrieval:
            has_docs = conversation_has_documents(
                user_id=request.user_id,
                conversation_id=conversation_id,
            )
            if has_docs:
                if scope == "document":
                    # Full coverage — bypass vector search entirely
                    context_chunks = retrieve_all_chunks_ordered(
                        user_id=request.user_id,
                        conversation_id=conversation_id,
                    )
                    context_chunks = _tag_chunks_with_pages(context_chunks)
                else:
                    # Specific topic — vector search as normal
                    top_k_map = {"low": 3, "medium": 7, "high": 20}
                    top_k = top_k_map.get(top_k_hint, 7)
                    query_embedding = pre_embedding if pre_embedding is not None else embed_query(q_text)
                    use_hybrid = bool(keywords) or query_type in ("formula", "definition", "list", "specific")
                    context_chunks = retrieve_chunks_smart(
                        query_embedding=query_embedding,
                        user_id=request.user_id,
                        conversation_id=conversation_id,
                        keywords=keywords if use_hybrid else None,
                        page_numbers=page_numbers if page_numbers else None,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                    )
                    print(f"[DEBUG-RETRIEVAL] pre-stream chunks retrieved: {len(context_chunks)}")
                    context_chunks = _tag_chunks_with_pages(context_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {str(e)}")

    # ── Event stream ──────────────────────────────────────────────────────────
    async def event_stream():
        active_chunks = context_chunks
        just_ingested_filenames: list = []

        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"

        # ── Auto-title for new conversations (fire-and-forget, zero latency) ──
        # The conversation was created with placeholder title "…" (shimmer in
        # sidebar). Now generate a real title in the background via a tiny LLM
        # call. The sidebar picks up the real title on its next refresh cycle.
        if not prior_messages:
            asyncio.ensure_future(_auto_title_conversation(
                conversation_id, request.user_id,
                request.message or "", intent,
                request.filename or "",
            ))

        # Blob RAG ingestion (when one or more files are sent WITH this message)
        # Runs INSIDE the stream so retrieval happens AFTER chunks are stored —
        # this is critical for page filtering to work on the first message with a file.

        # Build a unified list of ALL files to ingest from this message.
        # request.blob_url carries the real SAS for the primary file (File 1).
        # request.attachments carries real SAS blob_urls for ALL files (including File 1).
        # We deduplicate by name so File 1 is never OCR'd twice.
        files_to_ingest = []
        if request.blob_url and request.filename:
            files_to_ingest.append((request.blob_url, request.filename))
        if request.attachments:
            for att in request.attachments:
                if att.name != request.filename:   # skip File 1 — already added above
                    files_to_ingest.append((att.blob_url, att.name))

        if files_to_ingest:
            try:
                yield f"data: {json.dumps({'type': 'status', 'content': 'reading_document'})}\n\n"
                loop = asyncio.get_event_loop()
                all_ingested_chunks: list = []
                just_ingested_filenames: list = []

                create_index_if_not_exists()

                embed_semaphore = asyncio.Semaphore(10)  # shared across ALL files

                async def ingest_one_file(ingest_url: str, ingest_filename: str):
                    pages = await loop.run_in_executor(None, lambda u=ingest_url: extract_pages_from_url(u))
                    if not pages:
                        print(f"[WARN] No pages extracted from {ingest_filename} — skipping")
                        return None  # signals skip — handled in aggregation below

                    chunk_dicts         = chunk_by_paragraphs(pages)
                    chunks              = [c["text"] for c in chunk_dicts]
                    page_numbers_stored = [c["page_number"] for c in chunk_dicts]

                    async def embed_one(c):
                        async with embed_semaphore:
                            return await asyncio.get_event_loop().run_in_executor(None, lambda: embed_text(c))
                    embeddings = await asyncio.gather(*[embed_one(c) for c in chunks])

                    fid = str(uuid.uuid4())
                    store_chunks(
                        chunks=chunks, embeddings=embeddings,
                        user_id=request.user_id, conversation_id=conversation_id,
                        file_id=fid, filename=ingest_filename,
                        page_numbers=page_numbers_stored,
                    )
                    print(f"[DEBUG-INGESTION] Indexed {len(chunks)} chunks from '{ingest_filename}'")
                    return (chunks, ingest_filename)

                # All files OCR + embed + store in parallel
                # asyncio.gather preserves order — result[0] = file 1, result[1] = file 2, etc.
                ingest_results = await asyncio.gather(
                    *[ingest_one_file(url, name) for url, name in files_to_ingest]
                )

                # Aggregate in original upload order
                for result in ingest_results:
                    if result is None:
                        continue  # file was skipped (no pages extracted)
                    file_chunks, file_name = result
                    all_ingested_chunks.extend(file_chunks)
                    just_ingested_filenames.append(file_name)

                if all_ingested_chunks:
                    # ── Wait for Azure Search to index the chunks ─────────────────
                    # Azure Search upload_documents() returns when docs are accepted,
                    # NOT when they are searchable. Without this delay, retrieval
                    # returns empty and the LLM has no document context.
                    await asyncio.sleep(2)

                    # ── Filter to just-uploaded file when exactly one file ingested ──
                    post_filename_filter = (
                        just_ingested_filenames[0]
                        if len(just_ingested_filenames) == 1
                        else None
                    )

                    # ── Retrieval AFTER all files stored so every file's chunks are findable ──
                    if scope == "document":
                        # Full coverage — bypass vector search, fetch all chunks ordered
                        active_chunks = retrieve_all_chunks_ordered(
                            user_id=request.user_id,
                            conversation_id=conversation_id,
                            filename_filter=post_filename_filter,
                        )
                    else:
                        total = len(all_ingested_chunks)
                        tk = min(total, 20)

                        if request.message.strip():
                            post_q_text = request.message
                        elif topic:
                            post_q_text = topic
                        else:
                            post_q_text = all_ingested_chunks[0][:500] if all_ingested_chunks else "key concepts"

                        q_emb = embed_query(post_q_text)
                        use_hybrid_post = bool(keywords) or query_type in ("formula", "definition", "list", "specific")

                        active_chunks = retrieve_chunks_smart(
                            query_embedding=q_emb,
                            user_id=request.user_id,
                            conversation_id=conversation_id,
                            keywords=keywords if use_hybrid_post else None,
                            page_numbers=page_numbers if page_numbers else None,
                            top_k=tk,
                            use_hybrid=use_hybrid_post,
                            filename_filter=post_filename_filter,
                        )
                    print(f"[DEBUG-RETRIEVAL] post-ingestion chunks retrieved: {len(active_chunks)}")

                    # Sort by page and tag each chunk with [Page N]
                    active_chunks = _tag_chunks_with_pages(active_chunks)

                # Save a confirmation message listing ALL ingested files

                yield f"data: {json.dumps({'type': 'status', 'content': 'done'})}\n\n"
            except Exception as e:
                import traceback
                print(f"[ERROR] Ingestion failed: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'status', 'content': 'done'})}\n\n"

        # ── Dispatch ─────────────────────────────────────────────────────────
        # When exactly one file was just ingested in THIS message, pass its
        # filename so dispatch functions filter retrieval to the new file.
        # Multi-file uploads (len>1) → None → search all files as before.
        # No upload (len==0) → None → search all files as before.
        filename_hint = (
            just_ingested_filenames[0]
            if len(just_ingested_filenames) == 1
            else None
        )

        if intent == "quiz":
            async for evt in _dispatch_quiz(request.user_id, conversation_id, topic, num_questions, timer_seconds=timer_seconds, page_numbers=page_numbers, scope=scope, curriculum_context=_curr_context, provider=_provider, filename_hint=filename_hint):
                yield evt
            return

        if intent in ("flowchart", "mindmap"):
            dtype = "flowchart" if intent == "flowchart" else "diagram"
            async for evt in _dispatch_diagram(request.user_id, conversation_id, topic, dtype, prior_messages, raw_message=request.message, page_numbers=page_numbers, scope=scope, curriculum_context=_curr_context, provider=_provider, filename_hint=filename_hint):
                yield evt
            return

        if intent == "image":
            async for evt in _dispatch_image(request.user_id, conversation_id, topic, prior_messages, scope=scope, filename_hint=filename_hint):
                yield evt
            return

        if intent == "study_plan":
            tw = int(timeline_weeks) if timeline_weeks else 4
            hw = int(hours_per_week) if hours_per_week else 8
            async for evt in _dispatch_study_plan(request.user_id, conversation_id, topic, tw, hw, page_numbers=page_numbers, scope=scope, curriculum_context=_curr_context, provider=_provider, filename_hint=filename_hint):
                yield evt
            return

        if intent == "web_search":
            async for evt in _dispatch_web_search(
                user_id=request.user_id,
                conversation_id=conversation_id,
                query=request.message,
                prior_messages=prior_messages,
                has_attachment=bool(request.blob_url or request.attachments),
                pre_has_docs=_pre_has_docs,
            ):
                yield evt
            return

        # ── Regular chat ──────────────────────────────────────────────────────
        full_reply = ""
        try:
            for chunk in _provider.chat_stream(
                question=request.message,
                context_chunks=active_chunks,
                history=prior_messages,
                response_format=response_format,
                detail_level=detail_level,
                language_style=language_style,
                curriculum_context=_curr_context,
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                await asyncio.sleep(0)  # flush: yield event loop control after every token
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Mid-stream refusal fallback — if the safety message appeared after
        # some tokens were already streamed, save only the clean refusal to
        # Cosmos so conversation history stays correct.
        if _REFUSAL_MESSAGE in full_reply and full_reply.strip() != _REFUSAL_MESSAGE:
            full_reply = _REFUSAL_MESSAGE

        saved = await save_message(
            conversation_id=conversation_id,
            user_id=request.user_id,
            role="assistant",
            content=full_reply,
        )
        yield f"data: {json.dumps({'type': 'message_saved', 'message_id': saved['id']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Feature dispatch helpers ──────────────────────────────────────────────────

async def _dispatch_quiz(user_id, conversation_id, topic, num_questions, timer_seconds=None, page_numbers=None, scope=None, curriculum_context=None, provider=None, filename_hint=None):
    # provider defaults to the module-level import (server default) when not supplied.
    from app.services.ai_service import get_provider as _gp
    _p = provider or _gp(None)

    # ── Relevance threshold for scope=topic with a specific topic ─────────
    # Quiz uses use_hybrid=bool(topic) → when topic exists, hybrid is ON
    # → scores are RRF range (~0.01-0.03). Matches retrieve_chunks_hybrid.
    # Adjust this value if filtering is too aggressive or too loose.
    _RRF_THRESHOLD = 0.030

    try:
        if scope == "general":
            # General knowledge topic — skip document retrieval entirely.
            context_chunks = []
        elif scope in ("document", "page"):
            # User explicitly asked for document content or specific pages —
            # always retrieve, no score check.
            has_docs = conversation_has_documents(user_id=user_id, conversation_id=conversation_id)

            if has_docs:
                filenames = get_conversation_filenames(user_id=user_id, conversation_id=conversation_id)
                filename_filter = filename_hint or resolve_document_filter(topic or "", filenames)

                if scope == "document":
                    raw_chunks = retrieve_all_chunks_ordered(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        filename_filter=filename_filter,
                    )
                else:
                    q_emb = embed_query(topic or "key concepts and important topics")
                    raw_chunks = retrieve_chunks_smart(
                        query_embedding=q_emb,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        top_k=10,
                        use_hybrid=bool(topic),
                        keywords=[topic] if topic else None,
                        filename_filter=filename_filter,
                        page_numbers=page_numbers if page_numbers else None,
                    )
                context_chunks = [text for text, *_ in raw_chunks]
            else:
                context_chunks = []
        else:
            # scope == "topic" (or None/unknown — treat same as topic)
            has_docs = conversation_has_documents(user_id=user_id, conversation_id=conversation_id)

            if has_docs:
                filenames = get_conversation_filenames(user_id=user_id, conversation_id=conversation_id)
                filename_filter = filename_hint or resolve_document_filter(topic or "", filenames)

                q_emb = embed_query(topic or "key concepts and important topics")
                raw_chunks = retrieve_chunks_smart(
                    query_embedding=q_emb,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    top_k=10,
                    use_hybrid=bool(topic),
                    keywords=[topic] if topic else None,
                    filename_filter=filename_filter,
                    page_numbers=page_numbers if page_numbers else None,
                )

                if topic:
                    # Specific topic provided — check if doc content is relevant.
                    # If all scores below threshold, topic isn't in the doc →
                    # fall back to general knowledge.
                    relevant = [r for r in raw_chunks if r[3] >= _RRF_THRESHOLD]
                    if relevant:
                        context_chunks = [text for text, *_ in relevant]
                    else:
                        print(f"[QUIZ] All chunk scores below {_RRF_THRESHOLD} for topic '{topic}' — falling back to general knowledge")
                        context_chunks = []
                else:
                    # No specific topic — user wants quiz from doc content.
                    # Keep all chunks, no score filtering.
                    context_chunks = [text for text, *_ in raw_chunks]
            else:
                context_chunks = []

        result = _p.generate_quiz_questions(
            context_chunks=context_chunks,
            topic=topic or "",
            num_questions=num_questions,
            curriculum_context=curriculum_context,
        )

        # ── Safety sentinel check ─────────────────────────────────────────────
        # generate_quiz_questions() returns {"__refused__": True} when the model
        # detects the topic is harmful (covers chip path where classify_intent was skipped)
        if result.get("__refused__"):
            async for evt in _yield_refusal(conversation_id, user_id):
                yield evt
            return

        raw_questions = result["questions"]
        fun_fact = result["fun_fact"]

        quiz_id = str(uuid.uuid4())
        # Prefer the AI-generated title; fall back to cleaned topic string or "General Quiz"
        ai_title = (result.get("title") or "").strip()
        topic_label = ai_title or (topic or "General Quiz").strip()
        topic_label = re.sub(
            r'\b(and|or|the|a|an|for|of|in|on|with|about)\s*$',
            '', topic_label, flags=re.IGNORECASE,
        ).strip() or topic_label

        await save_quiz(
            user_id=user_id, quiz_id=quiz_id, topic=topic_label,
            questions=raw_questions, conversation_id=conversation_id,
            fun_fact=fun_fact,
        )

        q_for_history = [
            {"id": q["id"], "question": q["question"], "options": q["options"]}
            for q in raw_questions
        ]
        await save_message(
            conversation_id=conversation_id, user_id=user_id,
            role="assistant",
            content=json.dumps({
                "__type": "quiz", "quiz_id": quiz_id,
                "topic": topic_label, "submitted": False,
                "questions": q_for_history, "fun_fact": fun_fact,
                "timer_seconds": timer_seconds,
                "num_questions": num_questions,
            }),
        )

        yield f"data: {json.dumps({'type': 'quiz_result', 'data': {'quiz_id': quiz_id, 'topic': topic_label, 'questions': q_for_history, 'fun_fact': fun_fact, 'timer_seconds': timer_seconds, 'num_questions': num_questions}})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Quiz generation failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


async def _dispatch_diagram(user_id, conversation_id, topic, diagram_type, prior_messages, raw_message: str = "", page_numbers=None, scope=None, curriculum_context=None, provider=None, filename_hint=None):
    from app.services.ai_service import get_provider as _gp
    _p = provider or _gp(None)
    try:
        effective = topic or _p.infer_topic_from_messages(prior_messages) or "General Topic"

        msg_lower = (raw_message or "").lower()
        if any(w in msg_lower for w in ("circular", "circle", "cyclic", "cycle diagram", "loop diagram")):
            layout_hint = "circular"
        elif any(w in msg_lower for w in ("horizontal", "left to right", "lr", "sideways")):
            layout_hint = "horizontal"
        elif any(w in msg_lower for w in ("vertical", "top down", "td", "top to bottom")):
            layout_hint = "vertical"
        elif any(w in msg_lower for w in ("diagonal",)):
            layout_hint = "horizontal"
        else:
            layout_hint = None

        # ── Relevance threshold for scope=topic with a specific topic ─────
        # Diagram uses pure vector search (no hybrid) → cosine score range
        # 0.0-1.0.  Matches retrieve_chunks threshold.
        # Adjust this value if filtering is too aggressive or too loose.
        _COSINE_THRESHOLD = 0.65

        chunks = []
        try:
            if scope == "general":
                # General knowledge topic — skip document retrieval.
                pass  # chunks stays []
            elif scope in ("document", "page"):
                # User explicitly asked for document content or specific pages.
                has_docs = conversation_has_documents(user_id=user_id, conversation_id=conversation_id)

                if has_docs:
                    filenames = get_conversation_filenames(user_id=user_id, conversation_id=conversation_id)
                    filename_filter = filename_hint or resolve_document_filter(effective, filenames)

                    if scope == "document":
                        raw_chunks = retrieve_all_chunks_ordered(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            filename_filter=filename_filter,
                        )
                    else:
                        q_emb = embed_query(effective)
                        raw_chunks = retrieve_chunks_smart(
                            query_embedding=q_emb,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            top_k=8,
                            filename_filter=filename_filter,
                            page_numbers=page_numbers if page_numbers else None,
                        )
                    chunks = [text for text, *_ in raw_chunks]
            else:
                # scope == "topic" (or None/unknown)
                has_docs = conversation_has_documents(user_id=user_id, conversation_id=conversation_id)

                if has_docs:
                    filenames = get_conversation_filenames(user_id=user_id, conversation_id=conversation_id)
                    filename_filter = filename_hint or resolve_document_filter(effective, filenames)

                    q_emb = embed_query(effective)
                    raw_chunks = retrieve_chunks_smart(
                        query_embedding=q_emb,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        top_k=8,
                        filename_filter=filename_filter,
                        page_numbers=page_numbers if page_numbers else None,
                    )

                    if topic:
                        # Specific topic — check if doc content is relevant.
                        relevant = [r for r in raw_chunks if r[3] >= _COSINE_THRESHOLD]
                        if relevant:
                            chunks = [text for text, *_ in relevant]
                        else:
                            print(f"[DIAGRAM] All chunk scores below {_COSINE_THRESHOLD} for topic '{effective}' — falling back to general knowledge")
                            # chunks stays []
                    else:
                        # No specific topic — user wants from doc content.
                        chunks = [text for text, *_ in raw_chunks]
        except Exception:
            pass

        mermaid_code = _p.generate_mermaid(
            topic=effective,
            diagram_type=diagram_type,
            context_chunks=chunks,
            layout_hint=layout_hint,
            curriculum_context=curriculum_context,
        )

        # ── Safety sentinel check ─────────────────────────────────────────────
        # generate_mermaid() returns _REFUSAL_SENTINEL when it detects a harmful topic.
        # This is the PRIMARY guard for the chip path.
        if mermaid_code.strip() == _REFUSAL_SENTINEL:
            async for evt in _yield_refusal(conversation_id, user_id):
                yield evt
            return

        saved = await save_diagram(
            user_id=user_id, conversation_id=conversation_id,
            diagram_type=diagram_type, topic=effective, mermaid_code=mermaid_code,
        )
        await save_message(
            conversation_id=conversation_id, user_id=user_id,
            role="assistant",
            content=json.dumps({
                "__type": "diagram", "diagram_id": saved["diagram_id"],
                "type": saved["type"], "topic": saved["topic"],
                "mermaid_code": saved["mermaid_code"], "created_at": saved["created_at"],
            }),
        )
        yield f"data: {json.dumps({'type': 'diagram_result', 'data': saved})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Diagram generation failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


async def _dispatch_image(user_id, conversation_id, topic, prior_messages, scope=None, filename_hint=None):
    try:
        effective = topic or infer_topic_from_messages(prior_messages) or "General Topic"
        chunks = []
        try:
            if scope == "general":
                # General knowledge topic — skip document retrieval even if
                # docs exist in this conversation.
                pass  # chunks stays []
            else:
                has_docs = conversation_has_documents(user_id=user_id, conversation_id=conversation_id)

                if has_docs:
                    # Docs exist and scope is topic/page/document — retrieve context
                    q_emb = embed_query(effective)
                    chunks = retrieve_chunks(
                        query_embedding=q_emb, user_id=user_id,
                        conversation_id=conversation_id, top_k=3, score_threshold=0.5,
                        filename_filter=filename_hint,
                    )
                # else: no docs — chunks stays []
        except Exception:
            pass

        try:
            image_bytes = generate_image_bytes(topic=effective, context_chunks=chunks)
        except ValueError as ve:
            # generate_image() raises ValueError("__REFUSED__") for harmful topics
            if _REFUSAL_SENTINEL in str(ve):
                async for evt in _yield_refusal(conversation_id, user_id):
                    yield evt
                return
            raise

        blob_result = upload_generated_image_to_blob(image_bytes=image_bytes, topic=effective, user_id=user_id)
        saved = await save_image_diagram(
            user_id=user_id, conversation_id=conversation_id,
            topic=effective, image_url=blob_result["blob_url"],
        )
        await save_message(
            conversation_id=conversation_id, user_id=user_id,
            role="assistant",
            content=json.dumps({
                "__type": "image", "diagram_id": saved["diagram_id"],
                "type": "image", "topic": saved["topic"],
                "image_url": saved["image_url"], "created_at": saved["created_at"],
            }),
        )
        yield f"data: {json.dumps({'type': 'image_result', 'data': saved})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Image generation failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


async def _dispatch_study_plan(user_id, conversation_id, topic, timeline_weeks, hours_per_week, page_numbers=None, scope=None, curriculum_context=None, provider=None, filename_hint=None):
    from app.services.ai_service import get_provider as _gp
    _p = provider or _gp(None)
    try:
        plan = await create_study_plan(
            user_id=user_id, conversation_id=conversation_id,
            topic=topic or None, timeline_weeks=timeline_weeks,
            hours_per_week=hours_per_week, focus_days=None,
            page_numbers=page_numbers, scope=scope,
            curriculum_context=curriculum_context,
            model_provider=("gemini" if "gemini" in getattr(_p, "__name__", "") else "azure"),
            filename_hint=filename_hint,
        )

        # ── Safety sentinel check ─────────────────────────────────────────────
        # create_study_plan delegates to generate_study_plan() which returns
        # {"__refused__": True} for harmful topics (covers chip path)
        if isinstance(plan, dict) and plan.get("__refused__"):
            async for evt in _yield_refusal(conversation_id, user_id):
                yield evt
            return

        weeks_data = [
            {
                "week_number": w.get("week_number", 0),
                "start_date": w.get("start_date", ""),
                "end_date": w.get("end_date", ""),
                "tasks": w.get("tasks", []),
                "estimate_hours": w.get("estimate_hours"),
            }
            for w in plan.get("weeks", [])
        ]
        result = {
            "plan_id": plan["plan_id"], "title": plan.get("title", ""),
            "start_date": plan.get("start_date", ""), "end_date": plan.get("end_date", ""),
            "weeks": weeks_data, "summary": plan.get("summary", ""), "goal_saved": False,
        }
        await save_message(
            conversation_id=conversation_id, user_id=user_id,
            role="assistant",
            content=json.dumps({"__type": "study_plan", **result}),
        )
        yield f"data: {json.dumps({'type': 'study_plan_result', 'data': result})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Study plan generation failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


# ── _rag_from_doc ─────────────────────────────────────────────────────────────

async def _rag_from_doc(user_id, conversation_id, query, prior_messages, loop, update_message_id=None):
    """
    Condition A handler — answers exclusively from the student's indexed document.

    Called when a document exists in the conversation (either freshly uploaded
    in this message, or previously indexed). Never makes any web API calls.

    Latency profile:
        ~150ms : extract_document_context + embed_query run in parallel
        ~200ms : retrieve_chunks_smart / retrieve_all_chunks_ordered
        ~2000ms: chat_stream token generation
        ─────────────────────────────────────────────────────
        ~2350ms total (vs ~2850ms for the web path)

    Cosmos: saves as plain text — NOT as web_search_answer — so that the
    regenerate endpoint correctly re-runs RAG (not web search) on this message.
    """
    yield f"data: {json.dumps({'type': 'status', 'content': 'reading_document'})}\n\n"
    await asyncio.sleep(0)

    try:
        # Parallel: extract scope/page context AND embed the query simultaneously.
        # Both take ~150ms — running together costs the same as running one.
        doc_ctx, q_emb = await asyncio.gather(
            loop.run_in_executor(None, lambda: extract_document_context(query)),
            loop.run_in_executor(None, lambda: embed_query(query)),
        )

        scope        = doc_ctx.get("scope", "topic")
        page_numbers = doc_ctx.get("page_numbers") or []

        # Retrieve — strategy chosen by scope
        if scope == "document":
            # Full coverage: pure OData filter, no vector search, no top_k cap
            raw_chunks = await loop.run_in_executor(
                None,
                lambda: retrieve_all_chunks_ordered(
                    user_id=user_id,
                    conversation_id=conversation_id,
                ),
            )
        else:
            # Specific topic or page: hybrid vector + BM25 with optional page filter
            raw_chunks = await loop.run_in_executor(
                None,
                lambda: retrieve_chunks_smart(
                    query_embedding=q_emb,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    page_numbers=page_numbers if page_numbers else None,
                    top_k=10,
                    use_hybrid=True,
                    keywords=[query] if query else None,
                ),
            )

        context_chunks = _tag_chunks_with_pages(raw_chunks)

        yield f"data: {json.dumps({'type': 'status', 'content': 'done'})}\n\n"
        await asyncio.sleep(0)

        # Stream using the standard RAG chat_stream — same prompt as regular chat,
        # full conversation history preserved for follow-up continuity
        full_reply = ""
        try:
            for chunk in chat_stream(
                question=query,
                context_chunks=context_chunks,
                history=prior_messages,
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Mid-stream refusal fallback
        if _REFUSAL_MESSAGE in full_reply and full_reply.strip() != _REFUSAL_MESSAGE:
            full_reply = _REFUSAL_MESSAGE

        # Persist as plain text (not web_search_answer JSON) so regenerate
        # correctly re-runs RAG — not web search — on this message
        if update_message_id:
            await update_message_content(
                conversation_id=conversation_id,
                user_id=user_id,
                message_id=update_message_id,
                new_content=full_reply,
            )
            yield f"data: {json.dumps({'type': 'message_saved', 'message_id': update_message_id})}\n\n"
        else:
            saved_msg = await save_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=full_reply,
            )
            yield f"data: {json.dumps({'type': 'message_saved', 'message_id': saved_msg['id']})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Document retrieval failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


# ── _dispatch_web_search ──────────────────────────────────────────────────────

async def _dispatch_web_search(
    user_id,
    conversation_id,
    query,
    prior_messages,
    has_attachment: bool = False,
    pre_has_docs: bool | None = None,
    update_message_id=None,
    previous_message_content: Optional[str] = None,
):
    """
    Optimised web search dispatcher — minimum latency, 2 LLM calls total.

    ROUTING (zero extra cost):
        CHECK 1 — has_attachment field read (~0ms):
            File just ingested → Condition A immediately.

        CHECK 2 — conversation_has_documents result (~0ms if pre_has_docs supplied):
            pre_has_docs is pre-computed in parallel with save_message(user)
            before event_stream opens, so this check costs nothing here.
            Falls back to a live Azure Search ping only for regenerate path
            (where pre_has_docs=None because no parallel pre-computation ran).

    CONDITION A — Document exists:
        _rag_from_doc() → parallel embed_query + extract_document_context
        → retrieve → chat_stream. No SerpAPI, no YouTube, no image search.

    CONDITION B — No document:
        Phase 1 (parallel, ~150ms total):
            classify_search_intent(query)   LLM CALL 1 — routing + safety
            _fetch_web(query)               SerpAPI / YouTube / image_search
            Both fire simultaneously. SerpAPI no longer blocked by classifier.

        Phase 2 — Safety check on classify result:
            is_harmful=True  → discard web data, yield refusal
            is_harmful=False → proceed to LLM answer

        Phase 3 (~10-15s):
            chat_stream()   LLM CALL 2 — answer grounded in web results

    Net latency vs broken architecture: saves ~280ms + eliminates spike under load.

    previous_message_content: when supplied (regeneration path), the prior answer
        is included in the persisted payload via _build_web_search_answer_payload
        so that regen_versions history is preserved across page reloads.
    """
    try:
        loop = asyncio.get_event_loop()

        # Chip path can bypass classify_intent safety checks; block obvious
        # unsafe web queries before any network fetch or media retrieval.
        if _is_blocked_web_query(query):
            async for evt in _yield_refusal(conversation_id, user_id):
                yield evt
            return

        # ── CHECK 1: Zero-cost field check ────────────────────────────────────
        if has_attachment:
            async for evt in _rag_from_doc(
                user_id, conversation_id, query, prior_messages, loop, update_message_id
            ):
                yield evt
            return

        # ── CHECK 2: has_docs — use pre-computed result when available ────────
        # pre_has_docs is supplied by chat_message() which ran has_docs in
        # parallel with save_message(user) before event_stream opened.
        # For the regenerate path pre_has_docs=None, so we do a live ping here.
        yield f"data: {json.dumps({'type': 'status', 'content': 'searching_web'})}\n\n"
        await asyncio.sleep(0)

        if pre_has_docs is None:
            # Regenerate path or any caller that didn't pre-compute
            has_docs = await loop.run_in_executor(
                None,
                lambda: conversation_has_documents(
                    user_id=user_id, conversation_id=conversation_id
                ),
            )
        else:
            has_docs = pre_has_docs  # already known, zero cost

        if has_docs:
            async for evt in _rag_from_doc(
                user_id, conversation_id, query, prior_messages, loop, update_message_id
            ):
                yield evt
            return

        # ── CONDITION B: No document — parallel classify + SerpAPI ────────────
        # Both LLM classify and SerpAPI fire simultaneously.
        # classify was previously sequential (blocked SerpAPI) — now parallel.
        # After both resolve: check is_harmful before using web results.

        async def _classify():
            return await loop.run_in_executor(
                None, lambda: classify_search_intent(query)
            )

        async def _fetch_web_default():
            """Default text web search — runs in parallel with classifier."""
            try:
                return await loop.run_in_executor(
                    None, lambda: web_search(query, num_results=6)
                )
            except Exception as e:
                print(f"[WARN] Default web fetch failed: {e}")
                return []

        # Fire classify and a default text SerpAPI search simultaneously.
        # We always start with a text search; if classify says images/videos
        # we fire the correct call AFTER classify resolves (~150ms later,
        # while SerpAPI is still running its ~800ms round-trip).
        # This means for text queries (>80% of all queries) zero time is wasted.
        # For image/video queries we get a ~150ms head start on the text fetch
        # which we discard, then fire the correct call — net cost same as before.
        intent_result, default_sources = await asyncio.gather(
            _classify(),
            _fetch_web_default(),
        )

        result_type = intent_result.get("result_type", "text")
        web_query   = intent_result.get("web_query") or query
        is_harmful  = intent_result.get("is_harmful", False)

        # Safety gate — check BEFORE using any fetched data
        if is_harmful:
            async for evt in _yield_refusal(conversation_id, user_id):
                yield evt
            return

        # ── Stage 2: Resolve final web data ──────────────────────────────────────
        # default_sources (text search) was already fetched in parallel with
        # classify. For text queries (the majority) we use it directly — zero
        # extra wait. For image/video queries we fire the correct call now;
        # classify took ~150ms so SerpAPI has ~650ms left of its ~800ms trip.

        sources: list = []
        images:  list = []
        videos:  list = []

        if result_type == "text":
            # Use the already-fetched text results — no additional call needed
            # Filter out youtube.com entries: organic results sometimes rank youtube
            # pages which look like "video links" to the user even though they are
            # plain text citations. Respects explicit "no video" intent.
            sources = [
                r for r in default_sources
                if "youtube.com" not in r.get("source", "").lower()
                and "youtube.com" not in r.get("url", "").lower()
            ]

        elif result_type == "images":
            try:
                images = await loop.run_in_executor(
                    None, lambda: image_search(web_query, num_results=6)
                )
            except Exception as e:
                print(f"[WARN] Image search failed: {e}")

        elif result_type == "text_with_images":
            # Need both text and images; fire image search now,
            # reuse default_sources for text (already fetched)
            sources = default_sources
            try:
                images = await loop.run_in_executor(
                    None, lambda: image_search(web_query, num_results=6)
                )
            except Exception as e:
                print(f"[WARN] Image search failed: {e}")

        elif result_type == "videos":
            try:
                videos = await loop.run_in_executor(
                    None, lambda: youtube_search_api(web_query, num_results=5)
                )
            except Exception:
                try:
                    videos = await loop.run_in_executor(
                        None, lambda: youtube_search(web_query, num_results=5)
                    )
                except Exception as e:
                    print(f"[WARN] YouTube search failed: {e}")

        elif result_type == "text_with_videos":
            # Reuse default text fetch, fire YouTube in parallel
            async def _get_videos():
                try:
                    return await loop.run_in_executor(
                        None, lambda: youtube_search_api(web_query, num_results=5)
                    )
                except Exception:
                    try:
                        return await loop.run_in_executor(
                            None, lambda: youtube_search(web_query, num_results=5)
                        )
                    except Exception:
                        return []

            # text sources already in default_sources; get videos concurrently
            # with whatever minimal processing is left
            sources = default_sources
            videos  = await _get_videos()

        yield f"data: {json.dumps({'type': 'status', 'content': 'done'})}\n\n"
        await asyncio.sleep(0)

        # Stage 3: Build unified LLM prompt from web results only
        web_section = build_search_context(sources) if sources else ""

        if web_section:
            context_instruction = (
                "Answer exclusively from the web search results below. "
                "Do NOT add citation numbers like [1], [2] — sources are shown separately."
            )
        else:
            context_instruction = (
                "Neither web search results nor document chunks are available. "
                "Tell the student clearly that you could not find relevant information "
                "and suggest they rephrase their query."
            )

        # Base media/length instructions on ACTUAL fetched data — not result_type.
        # If a media search failed or returned empty, don't tell the LLM to mention
        # media that doesn't exist. This prevents phantom closing sentences.
        if images:
            media_instruction = (
                "At the very end of your response, add ONE short sentence like "
                "\"Here are some images of [topic]:\" — images are displayed below your text."
            )
        elif videos:
            media_instruction = (
                "At the very end of your response, add ONE short sentence like "
                "\"Here are some recommended videos on [topic]:\" — video cards are shown below.\n"
                "Do NOT include any URLs or markdown links in your response."
            )
        else:
            media_instruction = ""

        if result_type == "images" and images:
            length_instruction = (
                "Write ONE short sentence introducing the images "
                "(e.g. \"Here are some images of X:\"). Do NOT write an explanation."
            )
        elif result_type == "videos" and videos:
            length_instruction = (
                "Write ONE short sentence introducing the videos "
                "(e.g. \"Here are some videos about X:\"). Do NOT write an explanation."
            )
        else:
            length_instruction = (
                "Be comprehensive yet concise. "
                "Use markdown formatting (headings, bullet points) where helpful."
            )

        system_prompt = f"""You are StudyBuddy, a helpful educational assistant.

CURRENT QUESTION: "{query}"

{context_instruction}

{length_instruction}

{media_instruction}

{web_section}"""

        from app.services.ai_service import chat_stream as _ai_stream
        full_reply = ""
        try:
            for chunk in _ai_stream(
                question=query,
                context_chunks=[],
                history=[],
                system_prompt_override=system_prompt,
            ):
                full_reply += chunk
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # If the generated answer is a refusal, never attach web metadata.
        # Also catches mid-stream refusal fallback where _REFUSAL_MESSAGE
        # appears after some tokens were already streamed.
        if _is_refusal_text(full_reply) or (_REFUSAL_MESSAGE in full_reply):
            clean_refusal = _REFUSAL_MESSAGE
            if update_message_id:
                await update_message_content(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_id=update_message_id,
                    new_content=clean_refusal,
                )
                yield f"data: {json.dumps({'type': 'message_saved', 'message_id': update_message_id})}\n\n"
            else:
                saved_msg = await save_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role="assistant",
                    content=clean_refusal,
                )
                yield f"data: {json.dumps({'type': 'message_saved', 'message_id': saved_msg['id']})}\n\n"

            yield "data: [DONE]\n\n"
            return

        # Emit media SSE events below the text
        if images:
            yield f"data: {json.dumps({'type': 'web_search_images', 'data': images})}\n\n"
            await asyncio.sleep(0)

        if videos:
            yield f"data: {json.dumps({'type': 'web_search_videos', 'data': videos})}\n\n"
            await asyncio.sleep(0)

        if sources:
            yield f"data: {json.dumps({'type': 'web_search_sources', 'data': sources})}\n\n"
            await asyncio.sleep(0)

        # Persist as web_search_answer so regenerate re-runs web search on this message.
        # _build_web_search_answer_payload preserves regen_versions history when
        # previous_message_content is supplied (regeneration path).
        new_content = _build_web_search_answer_payload(
            query=query,
            answer=full_reply,
            sources=sources,
            images=images,
            videos=videos,
            previous_raw=previous_message_content,
        )

        if update_message_id:
            await update_message_content(
                conversation_id=conversation_id,
                user_id=user_id,
                message_id=update_message_id,
                new_content=new_content,
            )
            yield f"data: {json.dumps({'type': 'message_saved', 'message_id': update_message_id})}\n\n"
        else:
            saved_msg = await save_message(
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                content=new_content,
            )
            yield f"data: {json.dumps({'type': 'message_saved', 'message_id': saved_msg['id']})}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'Web search dispatch failed: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"


# ── GET /chat/history ─────────────────────────────────────────────────────────

@router.get("/history/{conversation_id}", response_model=ChatHistoryResponse)
async def chat_history(conversation_id: str):
    try:
        # Use get_conversation_full so we also retrieve model_provider in a
        # single Cosmos read — no extra round trip vs the old get_messages call.
        conv_data = await get_conversation_full(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {str(e)}")

    raw_messages = conv_data.get("messages", [])
    if not raw_messages:
        raise HTTPException(status_code=404, detail=f"No conversation found with id '{conversation_id}'.")

    messages = [
        ChatMessage(id=m["id"], role=m["role"], content=m["content"], timestamp=m["timestamp"])
        for m in raw_messages
    ]
    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        model_provider=conv_data.get("model_provider"),
    )


# ── GET /chat/conversations ───────────────────────────────────────────────────

@router.get("/conversations")
async def get_conversations(user_id: str = Query(...)):
    try:
        raw = await list_conversations(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversations: {str(e)}")

    conversations = []
    for conv in raw:
        # Explicit user rename always wins — check stored title FIRST
        if conv.get("title"):
            title = conv["title"]
        else:
            messages = conv.get("messages", [])
            first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if first_user_msg:
                raw_title = first_user_msg.get("content", "Untitled Chat")
                try:
                    if '"__type"' in raw_title:
                        parsed = json.loads(raw_title)
                        t = parsed.get("__type", "")
                        if t in ("user_with_attachments", "user_with_intent",
                                 "user_with_intent_and_attachments"):
                            raw_title = (
                                parsed.get("text", "") or
                                (parsed.get("attachments") or [{}])[0].get("name", "Untitled Chat")
                            )
                except Exception:
                    pass
                title = raw_title[:45] + ("..." if len(raw_title) > 45 else "")
            else:
                title = "New Conversation"
        conversations.append({
            "conversation_id": conv.get("conversation_id"),
            "title": title,
            "created_at": conv.get("created_at", ""),
            "updated_at": conv.get("updated_at", conv.get("created_at", "")),
            "starred": conv.get("starred", False),
        })

    return {"conversations": conversations}


# ── PATCH /chat/conversations/{id}/rename ─────────────────────────────────────

class RenameRequest(BaseModel):
    user_id: str
    title: str

@router.patch("/conversations/{conversation_id}/rename")
async def rename_conversation_endpoint(conversation_id: str, request: RenameRequest):
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    updated = await rename_conversation(
        conversation_id=conversation_id,
        user_id=request.user_id,
        new_title=request.title,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"ok": True}


# ── DELETE /chat/conversations/{id} ──────────────────────────────────────────

@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str, user_id: str = Query(...)):
    deleted = await delete_conversation(
        conversation_id=conversation_id,
        user_id=user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"ok": True}


# ── PATCH /chat/conversations/{id}/star ──────────────────────────────────────

class StarRequest(BaseModel):
    user_id: str
    starred: bool

@router.patch("/conversations/{conversation_id}/star")
async def star_conversation_endpoint(conversation_id: str, request: StarRequest):
    updated = await star_conversation(
        conversation_id=conversation_id,
        user_id=request.user_id,
        starred=request.starred,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"ok": True, "starred": request.starred}



@router.post("/translate")
async def translate_message(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        translated = translate_text(text=request.text, target_language=request.target_language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"translated_text": translated, "target_language": request.target_language}


# ── POST /chat/tts ────────────────────────────────────────────────────────────

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Strip code blocks and math expressions — Azure Speech chokes on both
    clean_text = re.sub(r"```[\s\S]*?```", "", text)   # fenced code
    clean_text = re.sub(r"\\\[[\s\S]*?\\\]", "", clean_text)  # \[...\] display math
    clean_text = re.sub(r"\\\([\s\S]*?\\\)", "", clean_text)  # \(...\) inline math
    clean_text = re.sub(r"\$\$[\s\S]*?\$\$", "", clean_text)    # $$...$$ display math
    clean_text = re.sub(r"\$[^\$\r\n]+\$", "", clean_text)       # $...$ inline math
    clean_text = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_text)
    clean_text = re.sub(r"\*(.*?)\*",     r"\1", clean_text)
    clean_text = re.sub(r"`(.*?)`",       r"\1", clean_text)
    clean_text = re.sub(r"#{1,6}\s",      "",    clean_text)
    clean_text = re.sub(r"[-*]\s",        "",    clean_text)
    clean_text = clean_text.strip()

    if not clean_text:
        raise HTTPException(status_code=400, detail="No speakable text after cleaning.")

    try:
        mp3_bytes = synthesize_speech(text=clean_text, language=request.language, voice_style=request.voice_style)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return StreamingResponse(
        io.BytesIO(mp3_bytes),
        media_type="audio/mpeg",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


# ── POST /chat/infer-topic ────────────────────────────────────────────────────

@router.post("/infer-topic")
async def infer_topic(request: InferTopicRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    try:
        topic = infer_topic_from_messages(request.messages)
        return {"topic": topic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic inference failed: {str(e)}")




# ── GET /chat/shared/{conversation_id} ───────────────────────────────────────

@router.get("/shared/{conversation_id}")
async def get_shared_conversation(conversation_id: str):
    try:
        conv_data = await get_conversation_full(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch shared conversation: {str(e)}")

    raw_messages = conv_data.get("messages", [])
    if not raw_messages:
        raise HTTPException(status_code=404, detail="Conversation not found or has no messages.")

    messages = [
        {"id": m["id"], "role": m["role"], "content": m["content"], "timestamp": m["timestamp"]}
        for m in raw_messages
    ]

    return {
        "conversation_id": conversation_id,
        "title": conv_data.get("title", "Shared Conversation"),
        "messages": messages,
    }
