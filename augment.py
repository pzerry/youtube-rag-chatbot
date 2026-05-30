def create_context(retrieved_docs):

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    return context