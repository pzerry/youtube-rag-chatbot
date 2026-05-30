from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

def generate_answer(
    context,
    question
):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_api_key
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer only from context.

Context:
{context}

Question:
{question}
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(
        final_prompt
    )

    return response.content