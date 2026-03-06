import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
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

def configurar_ferramenta_rag(api_key: str): 
    # 1. Limpa a chave (remove espaços ocultos e aspas residuais)
    chave_limpa = api_key.strip().strip("'").strip('"')
    
    # 2. Força a variável de ambiente (o Google exige isso nos bastidores)
    os.environ["GOOGLE_API_KEY"] = chave_limpa
    
    criar_documento_ficticio()
    
    loader = TextLoader(ARQUIVO_POLITICAS, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # 3. Usa o modelo mais recente e estável do Google para textos
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-001", 
        google_api_key=chave_limpa
    )
    
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    @tool
    def consultar_politicas_rh(query: str) -> str:
        """Busca informações oficiais no Código de Conduta e manuais de RH da empresa. 
        Use isso sempre que o líder perguntar sobre regras, demissões, assédio, horários ou processos formais."""
        documentos = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in documentos])
    
    return [consultar_politicas_rh]
