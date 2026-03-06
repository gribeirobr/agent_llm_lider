import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent import criar_grafo_agente

st.set_page_config(page_title="Mentor de Liderança AI", page_icon="👔")

st.title("👔 Mentor de Liderança AI")
st.markdown("Bem-vindo! Sou o assistente de RH e Liderança. Como posso ajudar na gestão do seu time hoje?")

try:
    api_key = st.secrets["API_KEY"]
except KeyError:
    st.error("Chave de API não encontrada! Configure os Secrets no Streamlit Cloud.")
    st.stop()

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

# FUNÇÃO NOVA: Descasca a resposta do Gemini para extrair só o texto limpo
def extrair_texto(conteudo):
    if isinstance(conteudo, str):
        return conteudo
    elif isinstance(conteudo, list):
        textos = []
        for bloco in conteudo:
            if isinstance(bloco, dict) and 'text' in bloco:
                textos.append(bloco['text'])
            elif isinstance(bloco, str):
                textos.append(bloco)
        return "\n".join(textos)
    return str(conteudo)

# Renderiza histórico limpando a formatação
for msg in st.session_state.messages:
    # Mostra apenas as mensagens do humano e da IA (esconde as chamadas internas de ferramentas)
    if msg.type in ["human", "ai"] and msg.content:
        with st.chat_message(msg.type):
            st.markdown(extrair_texto(msg.content))

if prompt := st.chat_input("Digite sua dúvida de liderança aqui..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    graph = obter_agente(api_key)
    
    with st.chat_message("ai"):
        with st.spinner("Analisando cenário e consultando políticas..."):
            inputs = {"messages": st.session_state.messages}
            config = {"configurable": {"thread_id": "sessao_1"}}
            
            response = graph.invoke(inputs, config=config)
            
            # Pega a última mensagem e extrai apenas o texto
            resposta_final = response["messages"][-1]
            texto_limpo = extrair_texto(resposta_final.content)
            
            # Mostra na tela perfeitamente formatado
            st.markdown(texto_limpo)
            
            st.session_state.messages = response["messages"]
