import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.config import MODEL_NAME, TEMPERATURE
from src.prompts import SYSTEM_PROMPT
from src.tools import configurar_ferramenta_rag

def criar_grafo_agente(api_key: str):
    """Monta e compila o fluxo do LangGraph."""
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 1. Carrega as ferramentas e o modelo
    tools = configurar_ferramenta_rag()
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    llm_with_tools = llm.bind_tools(tools)
    
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    
    # 2. Define o nó principal (Assistente)
    def assistente(state: MessagesState):
        mensagens = state["messages"]
        if not mensagens or not isinstance(mensagens[0], SystemMessage):
            mensagens = [sys_msg] + mensagens
        resposta = llm_with_tools.invoke(mensagens)
        return {"messages": [resposta]}
    
    # 3. Monta o Grafo
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", assistente)
    workflow.add_node("tools", ToolNode(tools)) 
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition) 
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
