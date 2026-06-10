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

# Lista de tus archivos específicos
ARCHIVOS_DOCUMENTOS = [
    "ANDINA TRAX SPM - Mod. de Capacitación - JUNIO 2026.pdf",
    "ANDINA TRAX Auto-Alma-Maxikiosco - Mod. de Capacitacion - JUNIO 2026.pdf"
]

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    gemini_api_key = st.text_input("Introduce tu Gemini API Key:", type="password")
    st.info("Este código procesará los manuales adjuntos de forma 100% gratuita usando Google Gemini.")

@st.cache_resource(show_spinner="Digitalizando manuales por goteo seguro (evitando límites de Google)...")
def inicializar_base_conocimientos(archivos, api_key):
    """Carga los PDFs, crea bloques grandes y los indexa uno a uno con pausas estratégicas."""
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
            st.warning(f"Archivo no encontrado: {archivo}")
            
    if not todos_los_documentos:
        return None

    # Fragmentos optimizados para reducir peticiones
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(todos_los_documentos)

    if not chunks:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Inicialización e indexación por goteo seguro
    vector_store = FAISS.from_documents([chunks[0]], embeddings)
    
    for chunk in chunks[1:]:
        time.sleep(1.2)
        vector_store.add_documents([chunk])
        
    return vector_store

# Verificar la API Key
if gemini_api_key:
    vector_store = inicializar_base_conocimientos(ARCHIVOS_DOCUMENTOS, gemini_api_key)
    
    if vector_store:
        # CEREBRO ACTUALIZADO AQUÍ: Se cambió a gemini-2.5-flash
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Historial de mensajes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Entrada del usuario
        if user_question := st.chat_input("¿Qué deseas saber sobre ANDINA TRAX?"):
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en los manuales..."):
                    # 1. Buscar las páginas más relevantes en los PDFs
                    docs_relevantes = vector_store.similarity_search(user_question, k=4)
                    
                    # 2. Unir el texto de los manuales
                    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
                    
                    # 3. Crear las instrucciones súper estrictas
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
                    
                    # 4. Obtener respuesta de Gemini
                    response = llm.invoke(instrucciones)
                    respuesta_final = response.content
                    
                    st.write(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                    
                    # Mostrar las fuentes de donde sacó la respuesta
                    with st.expander("Ver fuentes consultadas en el documento"):
                        for doc in docs_relevantes:
                            origen = doc.metadata.get('source', 'Desconocido')
                            pagina = doc.metadata.get('page', 0) + 1
                            st.write(f"• **Doc:** {origen} | **Página:** {pagina}")
    else:
        st.info("Asegúrate de que los archivos PDF estén en la misma carpeta.")
else:
    st.warning("Por favor, introduce tu Gemini API Key en la barra lateral para comenzar.")