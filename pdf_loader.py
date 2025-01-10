import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv()
# to create a new file named vectorstore in your current directory.
def load_knowledgeBase(uploaded_file=None):
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        api_key = os.getenv('OPENAI_API_KEY')
        
        # If a new file is uploaded, recreate the knowledge base
        if uploaded_file is not None:
            # Clear existing vector store
            if os.path.exists(DB_FAISS_PATH):
                import shutil
                shutil.rmtree(DB_FAISS_PATH)
            
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                # Create the vector store from uploaded PDF
                loader = PyPDFLoader("temp.pdf")
                documents = loader.load()
                
                # Add debug information
                #sl.info(f"Number of pages loaded: {len(documents)}")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Add debug information
                #sl.info(f"Number of text chunks created: {len(texts)}")
                
                # Create and save the FAISS index
                embeddings = OpenAIEmbeddings(api_key=api_key)
                db = FAISS.from_documents(texts, embeddings)
                
                # Create directory if it doesn't exist
                os.makedirs(DB_FAISS_PATH, exist_ok=True)
                
                # Save the vector store
                db.save_local(DB_FAISS_PATH)
                
                # Clean up temporary file
                os.remove("temp.pdf")
                
                sl.success("Knowledge base created successfully!")
                
            except Exception as e:
                sl.error(f"Error processing PDF: {str(e)}")
                return None
                
        elif os.path.exists(DB_FAISS_PATH):
            try:
                embeddings = OpenAIEmbeddings(api_key=api_key)
                db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                sl.success("Existing knowledge base loaded successfully!")
            except Exception as e:
                sl.error(f"Error loading existing knowledge base: {str(e)}")
                return None
        else:
            sl.error("No PDF file has been uploaded yet!")
            return None
            
        return db

def load_prompt():
        prompt = """ You are an expert advisor providing comprehensive information. Structure your response as follows:

        1. DIRECT ANSWER (4-5 sentences):
        Provide a clear, direct but more concise summarize answer to the question.

        2. KEY DETAILS:
        • Break down the most relevant information into bullet points
        • Include specific numbers, dates, and requirements where available
        • Highlight eligibility criteria and funding amounts if mentioned

        3. SUPPORTING EVIDENCE:
        • Quote specific passages from the source material using "quotes"
        • Include page numbers or section references when available
        • Clearly indicate if certain details are implied rather than explicitly stated

        4. PRACTICAL NEXT STEPS:
        • List concrete actions the reader can take
        • Include relevant links, contact points, or application procedures mentioned
        • Specify any deadlines or time-sensitive information

        5. ADDITIONAL CONTEXT:
        • Mention related funding programs or alternatives if relevant
        • Note any important caveats or conditions
        • Highlight any recent changes or updates mentioned in the document
        

        Given below is the context and question of the user.
        context = {context}
        question = {question}

        If the answer is not in the pdf at all, answer "I do not know."
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

def load_llm():
        api_key = os.getenv('OPENAI_API_KEY')
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, api_key=api_key)
        return llm

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def generate_top_5_questions(llm, text):
    prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an expert research analyst. 
        Focus on creating questions that:
        1. Cover critical funding requirements and eligibility
        2. Address practical application processes
        3. Reveal hidden opportunities or lesser-known benefits
        4. Explore strategic timing and preparation needs
        5. Uncover potential challenges and solutions
        
        Format each question to:
        - Be specific and actionable
        - Include relevant numbers or criteria when possible
        - Connect multiple concepts for deeper insights
        - Challenge common assumptions
        - Reveal strategic advantages
        
        Example style:
        - Instead of "What funding is available?", ask "Which EU funding programs offer the highest success rates for early-stage startups, and what unique eligibility criteria do they require?"
        - Instead of "How to apply?", ask "What are the three most critical preparation steps that successful applicants typically complete 6 months before submitting their EU funding application?"
        enerate questions that would uncover the most valuable insights about the given document. Make them specific and practical.
        Given the following text, generate 5 questions that a user might ask about this content. Return just the questions without numbers:
        {context}
        """
    )
    
    chain = LLMChain(prompt=prompt, llm=llm)
    questions = chain.run(context=text)
    cleaned_questions = [q.strip().lstrip('0123456789.)-] ') for q in questions.split('\n') if q.strip()]
    return cleaned_questions[:5]  # Get the first 5 questions

if __name__=='__main__':
        sl.header("Welcome to the 📝PDF bot")
        
        # Add file uploader
        uploaded_file = sl.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            sl.success("PDF file uploaded successfully!")
            with sl.spinner("Processing PDF..."):
                knowledgeBase = load_knowledgeBase(uploaded_file)
                
            if knowledgeBase is not None:
                sl.write("🤖 You can chat by Entering your queries")
                llm = load_llm()
                prompt = load_prompt()
                
                # Only show suggested questions if a file is uploaded
                sl.subheader("Top 5 Suggested Questions")
                suggested_questions = generate_top_5_questions(llm, uploaded_file)
                for question in suggested_questions:
                    sl.write(f"💬  {question}")
                
                query = sl.text_input('Enter some text')
        else:
            knowledgeBase = load_knowledgeBase()
            if knowledgeBase is not None:
                sl.write("🤖 You can chat by Entering your queries")
                llm = load_llm()
                prompt = load_prompt()
                query = sl.text_input('Enter some text')
        
        # Rest of the query processing code
        if 'query' in locals() and query:
            try:
                with sl.spinner("Searching for answer..."):
                    # Get similar documents
                    similar_embeddings = knowledgeBase.similarity_search(query)
                    
                    # Add debug information
                    sl.info(f"Found {len(similar_embeddings)} relevant document chunks")
                    
                    similar_embeddings = FAISS.from_documents(
                        documents=similar_embeddings, 
                        embedding=OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
                    )
                    
                    retriever = similar_embeddings.as_retriever()
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(query)
                    sl.write(response)
            except Exception as e:
                sl.error(f"Error processing query: {str(e)}")