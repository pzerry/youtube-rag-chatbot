from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
''' 
                                INDEXING
'''
############################## DOCUMENT INGESTION ##############################
video_id = "JMUxmLyrhSk"  
try: 
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id)
    transcript = " ".join([t.text for t in transcript_list])
#    print(transcript)  # Print the first 500 characters of the transcript
    
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
######################################## DOCUMENT SPLITTING ##############################
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([transcript])
#print((chunks))  # Print the first 500 characters of the first chunk
###################################### VECTOR STORE CREATION ##############################
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
##################### RETRIVAL ###############################
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

############ AUGMENTATION ##############################

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
prompt = PromptTemplate(
    template="""
    You are an helpful assistance. 
    Providing answer from transcript ONLY.
    If you dont know you say no idea but u will not hallucinate and say dont know.
    {context}
    Question:{question}
    """,
    input_variables = ['context','question']
)


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

########################## BUILDING CHAIN ##############################

parallel_chain = RunnableParallel(

    {
        "context": retriever | RunnableLambda(format_docs),

        "question": RunnablePassthrough()
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser
print(main_chain.invoke('What is the video about?'))