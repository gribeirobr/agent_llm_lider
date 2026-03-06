import os
from langchain_google_genai import ChatGoogleGenerativeAI # <-- GOOGLE AQUI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.prompts import SYSTEM_PROMPT
from src.tools import configurar_ferramenta_rag

def criar_grafo_agente(api_key: str):
    """Monta e compila o fluxo do LangGraph."""
    
    # 1. Avisa o sistema que a chave é do Google (Precisa vir antes de configurar o RAG)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 2. Carrega as ferramentas e o modelo
    tools = configurar_ferramenta_rag()
    
    # 3. Inicializa o LLM gratuito do Google Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)
    
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # 4. Define o nó principal (Assistente)
    def assistente(state: MessagesState):
        mensagens = state["messages"]
        if not mensagens or not isinstance(mensagens[0], SystemMessage):
            mensagens = [sys_msg] + mensagens
        resposta = llm_with_tools.invoke(mensagens)
        return {"messages": [resposta]}
    
    # 5. Monta o Grafo
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", assistente)
    workflow.add_node("tools", ToolNode(tools)) 
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition) 
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
