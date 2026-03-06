import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # <-- Novo import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from src.config import ARQUIVO_POLITICAS, DATA_DIR

# ... (Mantenha a função criar_documento_ficticio() igual) ...

def configurar_ferramenta_rag():
    criar_documento_ficticio()
    
    loader = TextLoader(ARQUIVO_POLITICAS, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # <-- Usa o modelo gratuito de embeddings do Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    @tool
    def consultar_politicas_rh(query: str) -> str:
        """Busca informações oficiais no Código de Conduta e manuais de RH da empresa. 
        Use isso sempre que o líder perguntar sobre regras, demissões, assédio, horários ou processos formais."""
        documentos = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in documentos])
    
    return [consultar_politicas_rh]
