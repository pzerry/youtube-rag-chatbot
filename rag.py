import os

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_community.vectorstores import (
    FAISS,
)

from langchain_core.prompts import (
    PromptTemplate,
)

from langchain_groq import ChatGroq

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)

from database_service import (
    save_video_metadata,
    get_video_metadata,
)

# --------------------------------------------------
# FAISS CACHE
# --------------------------------------------------

CACHE_DIR = "cache/faiss_indexes"
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------

PROMPT = PromptTemplate(
    template="""
You are a helpful assistant answering questions strictly based on a YouTube video transcript.

Rules:
- Answer ONLY from the transcript context provided.
- If the answer is not in the transcript, say:
  "This wasn't covered in the video."
- Never hallucinate or guess.
- Be concise and clear.

Previous conversation:
{history}

Transcript context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "history",
        "context",
        "question",
    ],
)

# --------------------------------------------------
# VIDEO ID EXTRACTION
# --------------------------------------------------


def extract_video_id(url_or_id: str) -> str:
    """
    Extract video ID from URL
    or return raw ID.
    """

    import re

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url_or_id,
        )

        if match:
            return match.group(1)

    if len(url_or_id) == 11:
        return url_or_id

    raise ValueError(
        f"Could not extract video ID from: {url_or_id}"
    )


# --------------------------------------------------
# CACHE PATH
# --------------------------------------------------


def get_cache_path(video_id: str) -> str:
    return os.path.join(
        CACHE_DIR,
        video_id,
    )


# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------


def load_or_build_vectorstore(
    video_id: str,
) -> tuple[FAISS, dict]:

    cache_path = get_cache_path(
        video_id
    )

    # --------------------------------------
    # CACHE HIT
    # --------------------------------------

    if os.path.exists(cache_path):

        print(
            f"[Cache HIT] Loading FAISS index for {video_id}"
        )

        vectorstore = FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        metadata = get_video_metadata(
            video_id
        )

        if not metadata:

            metadata = {
                "video_id": video_id,
                "chunk_count": 0,
                "transcript_length": 0,
                "cached": True,
            }

        metadata["cached"] = True

        return (
            vectorstore,
            metadata,
        )

    # --------------------------------------
    # CACHE MISS
    # --------------------------------------

    print(
        f"[Cache MISS] Building FAISS index for {video_id}"
    )

    try:

        ytt = YouTubeTranscriptApi()

        transcript_list = ytt.fetch(
            video_id
        )

        transcript = " ".join(
            [t.text for t in transcript_list]
        )

    except TranscriptsDisabled:

        raise ValueError(
            "Transcripts are disabled for this video."
        )

    except Exception as e:

        raise ValueError(
            f"Failed to fetch transcript: {str(e)}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    print("\n" + "=" * 50)
    print("VIDEO ID:", video_id)
    print("TRANSCRIPT LENGTH:", len(transcript))
    print("CHUNK COUNT:", len(chunks))
    print("=" * 50 + "\n")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vectorstore.save_local(
        cache_path
    )

    metadata = {
        "video_id": video_id,
        "chunk_count": len(chunks),
        "transcript_length": len(transcript),
        "cached": False,
    }

    print(
        f"Saving metadata -> "
        f"chunks={len(chunks)}, "
        f"transcript_length={len(transcript)}"
    )

    save_video_metadata(
        video_id=video_id,
        chunk_count=len(chunks),
        transcript_length=len(transcript),
    )

    saved = get_video_metadata(video_id)

    print("Saved metadata:", saved)

    return (
        vectorstore,
        metadata,
    )


# --------------------------------------------------
# HISTORY FORMATTER
# --------------------------------------------------


def format_history(
    history: list[dict],
) -> str:

    if not history:
        return "No previous conversation."

    lines = []

    for turn in history[-6:]:

        lines.append(
            f"Human: {turn['question']}"
        )

        lines.append(
            f"Assistant: {turn['answer']}"
        )

    return "\n".join(lines)


# --------------------------------------------------
# DOCUMENT FORMATTER
# --------------------------------------------------


def format_docs(
    docs,
) -> str:

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# LCEL CHAIN
# --------------------------------------------------


def build_chain(
    vectorstore: FAISS,
    history: list[dict],
):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    history_str = format_history(
        history
    )

    parallel_chain = RunnableParallel(
        {
            "context":
                retriever
                | RunnableLambda(
                    format_docs
                ),

            "question":
                RunnablePassthrough(),

            "history":
                RunnableLambda(
                    lambda _: history_str
                ),
        }
    )

    return (
        parallel_chain
        | PROMPT
        | llm
        | StrOutputParser()
    )


# --------------------------------------------------
# QA
# --------------------------------------------------


def answer_question(
    vectorstore: FAISS,
    question: str,
    history: list[dict],
) -> str:

    chain = build_chain(
        vectorstore,
        history,
    )

    return chain.invoke(
        question
    )
