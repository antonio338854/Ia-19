import streamlit as st
from ctransformers import AutoModelForCausalLM
import time

# 1. Configuração da Página (Mobile First)
st.set_page_config(page_title="IA Local Mobile", page_icon="🤖")

st.title("🤖 IA Rodando 'Localmente'")
st.caption("Este chatbot roda 100% no servidor, sem APIs pagas. Modelo: TinyLlama.")

# 2. Carregamento do Modelo (Cacheado para não travar)
# O segredo do especialista: Usamos @st.cache_resource para carregar a IA na memória só uma vez!
@st.cache_resource
def carregar_ia():
    with st.spinner('Baixando e carregando o cérebro da IA (isso acontece só na 1ª vez)...'):
        # Usamos o TinyLlama quantizado. Pequeno, rápido e roda na CPU grátis.
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_type="llama",
            gpu_layers=0 # Força rodar na CPU
        )
    return llm

# Tenta carregar. Se der erro de memória, avisamos.
try:
    llm = carregar_ia()
except Exception as e:
    st.error(f"Erro ao carregar modelo: {e}")
    st.stop()

# 3. Histórico do Chat (Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou uma IA rodando direto no Python. Como posso ajudar?"}]

# 4. Exibir mensagens anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Lógica de Envio e Resposta
if prompt := st.chat_input("Digite sua mensagem..."):
    # Mostra mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Gera resposta da IA
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # Construção do prompt no formato que o Llama entende
        # Isso é cirúrgico para ele não se perder
        prompt_formatado = f"<|system|>\nVocê é um assistente útil.<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Geração do texto
        with st.spinner('Pensando...'):
            try:
                for text in llm(prompt_formatado, stream=True, max_new_tokens=256):
                    full_response += text
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                st.error("A IA cansou (Memória cheia). Tente recarregar.")
    
    # Salva resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.info("Criado pelo Especialista Mobile (60 anos de XP).")
st.sidebar.warning("Nota: Como roda em CPU, pode ser um pouco mais lento que o GPT, mas é 100% seu!")
