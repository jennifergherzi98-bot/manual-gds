import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Configuración de la página de Streamlit
st.set_page_config(page_title="Asistente Virtual - ANDINA TRAX", page_icon="🤖")
st.title("📚 Consultor de Documentación ANDINA TRAX (Gemini)")
st.write("Haz preguntas sobre los módulos de capacitación. Las respuestas se basarán **únicamente** en los archivos provistos.")

# Obtener la ruta exacta de la carpeta actual de forma dinámica
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lista de tus archivos con los nombres exactos de tu GitHub
ARCHIVOS_DOCUMENTOS = [
    os.path.join(BASE_DIR, "ANDINA TRAX SPM-Mod. de Capacitación - JUNIO 2026.pdf"),
    os.path.join(BASE_DIR, "ANDINA TRAX Auto-Alma-Maxikiosco - Mod. de Capacitación - JUNIO 2026.pdf")
]

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    gemini_api_key = st.text_input("Introduce tu Gemini API Key:", type="password")
    st.info("Este código procesará los manuales adjuntos de forma 100% gratuita usando Google Gemini.")

@st.cache_resource(show_spinner="Digitalizando manuales en bloques seguros para tu clave gratuita...")
def inicializar_base_conocimientos(archivos, api_key):
    if not api_key:
        return None
    
    os.environ["GOOGLE_API_KEY"] = api_key
    todos_los_documentos = []
    
    for archivo in archivos:
        if os.path.exists(archivo):
            loader = PyPDFLoader(archivo)
            paginas = loader.load()
            todos_los_documentos.extend(paginas)
        else:
            st.warning(f"Archivo no encontrado: {os.path.basename(archivo)}")
            
    if not todos_los_documentos:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(todos_los_documentos)
    
    if not chunks:
        return None

    # Cambiamos al nuevo modelo oficial y estable de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    try:
        # Inicializamos el contenedor con los primeros 5 fragmentos
        batch_size = 5
        primer_batch = chunks[:batch_size]
        vector_store = FAISS.from_documents(primer_batch, embeddings)
        
        # Marcador visual para monitorear la carga en la pantalla
        estado_carga = st.empty()
        
        # Enviamos el resto en tandas de 5 fragmentos, esperando 5 segundos entre cada una
        for i in range(batch_size, len(chunks), batch_size):
            estado_carga.text(f"⏳ Procesando fragmentos {i} de {len(chunks)}...")
            time.sleep(5.0)  # Pausa estratégica para cumplir con el límite de la API gratuita
            batch = chunks[i:i+batch_size]
            vector_store.add_documents(batch)
            
        estado_carga.empty()
        return vector_store
    except Exception as e:
        st.error(f"Error de procesamiento con Google Gemini: {e}")
        return None

# --- CONTROL DEL FLUJO PRINCIPAL ---
if gemini_api_key:
    # Quitamos espacios accidentales que se puedan colar al pegar la clave
    api_key_limpia = gemini_api_key.strip()
    vector_store = inicializar_base_conocimientos(ARCHIVOS_DOCUMENTOS, api_key_limpia)
    
    if vector_store:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if user_question := st.chat_input("¿Qué deseas saber sobre ANDINA TRAX?"):
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en los manuales..."):
                    docs_relevantes = vector_store.similarity_search(user_question, k=4)
                    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
                    
                    instrucciones = (
                        "Eres un asistente experto en los módulos de capacitación de ANDINA TRAX.\n"
                        "Tu tarea es responder las preguntas utilizando ÚNICAMENTE el contexto de abajo.\n"
                        "REGLAS CRÍTICAS:\n"
                        "1. Si la respuesta no está explícitamente en el contexto, responde exactamente: "
                        "'Lo siento, no encuentro esa información en los manuales de capacitación provistos.'\n"
                        "2. No inventes ni uses conocimientos externos.\n\n"
                        f"CONTEXTO DE LOS MANUALES:\n{contexto}\n\n"
                        f"PREGUNTA DEL USUARIO: {user_question}"
                    )
                    
                    response = llm.invoke(instrucciones)
                    respuesta_final = response.content
                    
                    st.write(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                    
                    with st.expander("Ver fuentes consultadas en el documento"):
                        for doc in docs_relevantes:
                            origen = os.path.basename(doc.metadata.get('source', 'Desconocido'))
                            pagina = doc.metadata.get('page', 0) + 1
                            st.write(f"• **Doc:** {origen} | **Página:** {pagina}")
    else:
        st.info("Asegúrate de que los archivos PDF estén en la misma carpeta.")
else:
    st.warning("Por favor, introduce tu Gemini API Key en la barra lateral para comenzar.")
