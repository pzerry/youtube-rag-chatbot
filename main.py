from indexing import create_vectorstore
from retriever import get_retriever
from augment import create_context
from generate import generate_answer


video_id = "JMUxmLyrhSk"

question = "What is perceptron?"


vectorstore = create_vectorstore(
    video_id
)

retriever = get_retriever(
    vectorstore
)

docs = retriever.invoke(
    question
)

context = create_context(
    docs
)

answer = generate_answer(
    context,
    question
)

print(answer)