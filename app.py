import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent import criar_grafo_agente

st.set_page_config(page_title="Mentor de Liderança AI", page_icon="👔")

st.title("👔 Mentor de Liderança AI")
st.markdown("Bem-vindo! Sou o assistente de RH e Liderança. Como posso ajudar na gestão do seu time hoje?")

# 1. PEGA A CHAVE DIRETAMENTE DOS SEGREDOS DO STREAMLIT
try:
    api_key = st.secrets["API_KEY"]
except KeyError:
    st.error("Chave de API não encontrada! Configure os Secrets no Streamlit Cloud.")
    st.stop()

# 2. A BARRA LATERAL FICA SÓ COM AS DICAS (Sem campo de senha)
with st.sidebar:
    st.header("💡 Dicas de Uso")
    st.markdown("- *'Quero demitir um analista por baixa performance, qual o processo?'*")
    st.markdown("- *'Como dar um feedback sobre atrasos sem desmotivar a pessoa?'*")
    st.markdown("- *'Um colaborador relatou uma piada racista no time. O que eu faço?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def obter_agente(chave_api):
    return criar_grafo_agente(chave_api)

for msg in st.session_state.messages:
    if msg.type in ["human", "ai"] and msg.content:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

if prompt := st.chat_input("Digite sua dúvida de liderança aqui..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Passamos a chave que veio lá do st.secrets
    graph = obter_agente(api_key)
    
    with st.chat_message("ai"):
        with st.spinner("Analisando cenário..."):
            inputs = {"messages": st.session_state.messages}
            config = {"configurable": {"thread_id": "sessao_1"}}
            
            response = graph.invoke(inputs, config=config)
            resposta_final = response["messages"][-1]
            
            st.markdown(resposta_final.content)
            st.session_state.messages = response["messages"]
