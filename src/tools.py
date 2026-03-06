import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool # <-- Usaremos isso no lugar da função pronta
from src.config import ARQUIVO_POLITICAS, DATA_DIR

def criar_documento_ficticio():
    """Garante que a pasta 'data' e o arquivo de teste existam."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(ARQUIVO_POLITICAS):
        conteudo = """
        CÓDIGO DE CONDUTA E POLÍTICAS DE RH - EMPRESA EXEMPLO S/A
        
        Capítulo 1: Resolução de Conflitos Internos
        Líderes devem atuar como os primeiros mediadores em conflitos de equipe. A abordagem obrigatória é a escuta ativa e o uso da metodologia de Feedback SBI (Situação, Comportamento, Impacto). Conflitos puramente operacionais devem ser resolvidos na área.
        
        Capítulo 2: Tolerância Zero (Assédio e Discriminação)
        Nossa empresa tem tolerância zero para assédio moral, sexual, racismo, homofobia ou qualquer tipo de discriminação. Caso um líder presencie ou receba denúncia deste tipo, ele ESTÁ PROIBIDO de tentar resolver sozinho. O líder deve encaminhar o caso imediatamente para o comitê de ética (etica@empresa.com) e para o BP de RH responsável.
        
        Capítulo 3: Políticas de Home Office e Horários
        O modelo de trabalho é híbrido (3 dias no escritório, 2 em casa). Atrasos de até 15 minutos são tolerados sem necessidade de justificativa formal. Atrasos recorrentes devem ser tratados com feedback de alinhamento pelo gestor direto.
        
        Capítulo 4: Desligamentos e Baixa Performance
        Antes de qualquer desligamento por baixa performance, o colaborador deve passar por um Plano de Recuperação de Performance (PIP) com duração mínima de 30 dias. O gestor deve documentar feedbacks semanais durante este período.
        """
        with open(ARQUIVO_POLITICAS, "w", encoding="utf-8") as f:
            f.write(conteudo)

def configurar_ferramenta_rag():
    """Cria a ferramenta de busca vetorial que o agente vai usar."""
    criar_documento_ficticio()
    
    loader = TextLoader(ARQUIVO_POLITICAS, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # CRIANDO A FERRAMENTA MANUALMENTE (À PROVA DE FALHAS)
    @tool
    def consultar_politicas_rh(query: str) -> str:
        """Busca informações oficiais no Código de Conduta e manuais de RH da empresa. 
        Use isso sempre que o líder perguntar sobre regras, demissões, assédio, horários ou processos formais."""
        
        # O retriever busca os trechos mais relevantes
        documentos = retriever.invoke(query)
        
        # Junta os textos encontrados e devolve para o Agente ler
        return "\n\n".join([doc.page_content for doc in documentos])
    
    return [consultar_politicas_rh]
