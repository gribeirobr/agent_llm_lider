import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent import criar_grafo_agente

st.set_page_config(page_title="Mentor de Liderança AI", page_icon="👔")

st.title("👔 Mentor de Liderança AI")
st.markdown("Bem-vindo! Sou o assistente de RH e Liderança. Como posso ajudar na gestão do seu time hoje?")

with st.sidebar:
    st.header("Configurações")
    api_key = st.text_input("Insira sua OpenAI API Key", type="password")
    st.markdown("---")
    st.markdown("💡 **Exemplos:**")
    st.markdown("- *'Quero demitir um analista por baixa performance, qual o processo?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Evita recriar o grafo a cada interação pesada, usando cache no Streamlit
@st.cache_resource(show_spinner=False)
def obter_agente(chave_api):
    return criar_grafo_agente(chave_api)

# Renderiza histórico
for msg in st.session_state.messages:
    if msg.type in ["human", "ai"] and msg.content:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

# Interação com o usuário
if prompt := st.chat_input("Digite sua dúvida de liderança aqui..."):
    if not api_key:
        st.warning("Por favor, insira sua OpenAI API Key na barra lateral.")
        st.stop()
        
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    graph = obter_agente(api_key)
    
    with st.chat_message("ai"):
        with st.spinner("Analisando cenário..."):
            inputs = {"messages": st.session_state.messages}
            config = {"configurable": {"thread_id": "sessao_1"}}
            
            response = graph.invoke(inputs, config=config)
            resposta_final = response["messages"][-1]
            
            st.markdown(resposta_final.content)
            st.session_state.messages = response["messages"]
