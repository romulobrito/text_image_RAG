import streamlit as st
from PIL import Image
import time
from rag_utils import (
    preprocess_query_chain, 
    router_chain1, 
    router_chain2,
    retrieval_chain, 
    qa_chain, 
    eval_chain, 
    external_search_chain, 
    get_image_width,
    process_and_add_pdf,
    describe_and_rank_images
)

# Configuração da página - DEVE ser a primeira chamada Streamlit
st.set_page_config(
    page_title="Assistente de Moldes", 
    page_icon="🏭", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Constantes
DEFAULT_PDF_PATH = r"/home/romulobrito/projetos/image-RAG/text_image_rag/Injection Mold Design Handbook - preview-9781569908167_A42563111.pdf"
LOGO_PATH = r"/home/romulobrito/projetos/image-RAG/text_image_rag/logo_7d.jpg"

# ====================================================================
# Streamlit utils
# ====================================================================
def text_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk
        container.markdown(result, unsafe_allow_html=True)
    return container

def get_stream(text):
    for chunk in text.split():
        chunk = f"<span style='margin-right: 5px; color: gray;'>{chunk}</span>"
        time.sleep(0.03)
        yield chunk

def display_progress_message(text):
    container = write_stream(stream=get_stream(text))
    return container

def send_button_ques(question):
    st.session_state.disabled = True
    st.session_state['button_question'] = question

# ====================================================================
# Streamlit chat
# ====================================================================
system_prompt = """Você é um assistente especializado em moldes de injeção plástica, 
                  capaz de fornecer informações técnicas detalhadas e práticas.
                  Responda com exemplos específicos e detalhes técnicos para fornecer 
                  uma resposta abrangente sobre design, manufatura e boas práticas em moldes de injeção."""

questions = [
    'Boas práticas para construção de peças por injeção plástica?',
    'Mostre-me figuras sobre boas práticas na construção de moldes por injeção plástica.',
    'Quais são os principais sistemas de refrigeração em moldes?',
    'Como evitar defeitos comuns em peças injetadas?',
    'Explique o sistema de extração em moldes de injeção.'
]

def main():
    # Barra lateral para histórico
    with st.sidebar:
        st.image(LOGO_PATH, width=300)
        st.header("📝 Histórico de Conversas")
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            for idx, message in enumerate(st.session_state.messages[1:], 1):
                with st.container():
                    if message["role"] == "user":
                        st.markdown(f"👤 **Você**: {message['content'][:100]}...")
                    else:
                        st.markdown(f"🤖 **Assistente**: {message['content'][:100]}...")
                    st.divider()
        else:
            st.info("Nenhuma conversa anterior.")
            
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.messages = [{"role": "assistant", "content": 'Como posso ajudar você com informações sobre moldes de injeção plástica?'}]
            st.rerun()
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Bem-vindo ao Assistente de Moldes de Injeção!")
        st.write("🏭 Você pode interagir digitando suas perguntas sobre moldes de injeção plástica.")
        
    with col2:
        uploaded_file = st.file_uploader("Carregue um arquivo PDF 📝", type=["pdf"])
        knowledgeBase = None

        if uploaded_file is not None:
            knowledgeBase = process_and_add_pdf(uploaded_file=uploaded_file)
            st.success("Arquivo PDF carregado e processado com sucesso!")
        
        if knowledgeBase is None:
            st.info("Nenhum arquivo PDF carregado. Usando arquivo PDF padrão.")
            knowledgeBase = process_and_add_pdf(pdf_path=DEFAULT_PDF_PATH)

    with st.expander("📌 Dicas para melhores resultados", expanded=False):
        st.write("- Use termos técnicos precisos")
        st.write("- Faça perguntas específicas sobre aspectos do molde")
        st.write("- Inclua o contexto necessário em sua pergunta")

    st.divider()
    
    if "messages" not in st.session_state:
        first_message = 'Como posso ajudar você com informações sobre moldes de injeção plástica?'
        st.session_state.messages = [{"role": "assistant", "content": first_message}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if "disabled" not in st.session_state:
        st.session_state.disabled = False
        buttons = st.container()
        for q in questions:
            buttons.button(
                label=q, 
                on_click=send_button_ques, 
                args=[q], 
                disabled=st.session_state.disabled
            )

    # Handling user input and generating a response
    if user_input := (st.chat_input("You:", key="user_input") or st.session_state.get("button_question")):
        start_time = time.time()
        
        if "progress_placeholders" not in st.session_state:
            st.session_state.progress_placeholders = []
        else:
            for placeholder in st.session_state.progress_placeholders:
                placeholder.empty()
            st.session_state.progress_placeholders.clear()

        if "assistant_response_placeholder" in st.session_state:
            st.session_state.assistant_response_placeholder.empty()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        st.info("🕒 Iniciando processamento da consulta...")
        chain_input = {
            "system_prompt": system_prompt, 
            "chat_history": st.session_state.messages, 
            "question": user_input
        }

        with st.spinner("Thinking..."):
            st.info("🔄 Pré-processando consulta e histórico...")
            preprocess_output = preprocess_query_chain.invoke(chain_input)
            preprocess_progress_placeholder = display_progress_message("Condensed chat history and incorporated into user query.")
            st.session_state.progress_placeholders.append(preprocess_progress_placeholder)

            st.info("🔍 Classificando tipo de consulta (Fase 1)...")
            router_output1 = router_chain1.invoke(preprocess_output)
            classification1 = router_output1['classification1'].datasources
            router_progress_placeholder1 = display_progress_message(f"Agent classified query as: {classification1}")
            st.session_state.progress_placeholders.append(router_progress_placeholder1)

            st.info("🔍 Classificando tipo de consulta (Fase 2)...")
            router_output2 = router_chain2.invoke(router_output1)
            classification2 = router_output2['classification2'].datasources
            router_progress_placeholder2 = display_progress_message(f"Agent classified query as: {classification2}")
            st.session_state.progress_placeholders.append(router_progress_placeholder2)

            st.info("📚 Recuperando documentos relevantes...")
            retrieval_output = retrieval_chain.invoke(router_output2)
            retrieval_progress_placeholder = display_progress_message("Retrieved relevant documents. Generating initial answer...")
            st.session_state.progress_placeholders.append(retrieval_progress_placeholder)

            st.info("💭 Gerando resposta inicial...")
            qa_output = qa_chain.invoke(retrieval_output)
            qa_progress_placeholder = display_progress_message("Initial answer generated.")
            st.session_state.progress_placeholders.append(qa_progress_placeholder)

            st.info("⚖️ Avaliando qualidade da resposta...")
            eval_output = eval_chain.invoke(qa_output)
            eval_result = eval_output['eval_result']
            
            if eval_result == "Y":
                eval_progress_placeholder = display_progress_message("✅ Resposta avaliada como satisfatória.")
            else:
                eval_progress_placeholder = display_progress_message("🔄 Resposta insatisfatória. Iniciando busca externa...")
            st.session_state.progress_placeholders.append(eval_progress_placeholder)

            assistant_response = external_search_chain.invoke(eval_output)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            st.session_state.assistant_response_placeholder = st.empty()
            with st.session_state.assistant_response_placeholder.container():
                with st.chat_message("assistant"):
                    st.write(assistant_response)
                    if eval_result == "Y":
                        try:
                            st.info("🖼️ Buscando imagens relevantes...")
                            image_dir = "/home/romulobrito/projetos/image-RAG/text_image_rag/extracted_images/marco"
                            ranked_images = describe_and_rank_images(user_input, assistant_response, image_dir, n=3)
                            
                            if ranked_images:
                                st.success(f"✨ Encontradas {len(ranked_images)} imagens relevantes!")
                                cols = st.columns(min(len(ranked_images), 3))
                                for idx, (image_path, description, relevance) in enumerate(ranked_images):
                                    try:
                                        with cols[idx]:
                                            st.image(image_path, use_column_width=True)
                                            st.caption(f"Imagem {idx + 1}")
                                            st.markdown(f"""
                                            **Relevância**: {relevance:.2%}
                                            
                                            **Descrição Técnica**:
                                            {description}
                                            """)
                                    except Exception as e:
                                        st.error(f"❌ Erro ao carregar imagem {image_path}: {e}")
                            else:
                                st.warning("⚠️ Não foram encontradas imagens relevantes para esta consulta.")
                        except Exception as e:
                            st.error(f"❌ Erro ao processar imagens: {e}")

        end_time = time.time()
        latency = end_time - start_time
        st.info(f"⏱️ Tempo total de processamento: {latency:.2f} segundos")

if __name__ == "__main__":
    main()