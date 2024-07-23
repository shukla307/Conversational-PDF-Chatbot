from PyPDF2 import PdfReader, PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import io
import os

from fastapi import FastAPI, File, UploadFile,HTTPException

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

app = FastAPI()



def extract_text_from_pdf(pdf_bytes): 
    if type(pdf_bytes) is not io.BufferedReader: 
        # print(type(pdf_bytes) )
        pdf_bytes = io.BytesIO(pdf_bytes)  
    pdf_reader = PdfFileReader(pdf_bytes)
    text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text += page.extract_text()
    return text


class PDF_Chatbot():

    def __init__(self) -> None:
        load_dotenv()
        # spilit into chuncks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # create embedding
        self.embeddings = HuggingFaceEmbeddings()

        self.llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0.1,"max_length":128})
        self.chain = load_qa_chain(self.llm,chain_type="stuff")

    def update_knowledge_base(self,text):        
        self.chunks = self.text_splitter.split_text(text)
        self.knowledge_base = FAISS.from_texts(self.chunks,self.embeddings)

    def run_query(self,query):
        docs = self.knowledge_base.similarity_search(query)
        response = self.chain.run(input_documents=docs,question=query)
        # print("\n\nresponse: ",response)
        return response

pdf_chatbot  =  PDF_Chatbot()
        
@app.get("/")
async def root():
    return {"message": "Welcome to a PDF ChatBot!"}



@app.post("/uploadfile/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    try:
        # print(type(file))
        contents = await file.read()
        text_content = extract_text_from_pdf(contents)
        pdf_chatbot.update_knowledge_base(text=text_content)
        return {"filename": file.filename, "content": text_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat/")
async def chat_with_pdf(query: str):
    response  = pdf_chatbot.run_query(query)
    print("\n\nresponse: ",response)
    return {"response": response}
