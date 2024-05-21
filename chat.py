import streamlit as st
import openai
from dotenv import load_dotenv
import nltk
import os
import tempfile
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import PyPDF2
import time
from gtts import gTTS
import io

# Configuración de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Función para cargar el texto del PDF
def extraer_texto_pdf(archivo):
    texto = ""
    if archivo:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(archivo.read())
            temp_file_path = temp_file.name
        with open(temp_file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                texto += reader.pages[page].extract_text()
        os.unlink(temp_file_path)
    return texto

# Función para preprocesar texto
def preprocesar_texto(texto):
    tokens = word_tokenize(texto, language='spanish')
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stopwords_es = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stopwords_es]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Cargar la clave API desde el archivo .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Función para obtener respuesta de OpenAI usando el modelo GPT y convertir a audio
def obtener_respuesta(pregunta, texto_preprocesado, modelo, temperatura=0.5):
    try:
        response = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Actua como Galatea la asistente de la clinica Odontologica OMARDENT y resuelve las inquietudes"},
                {"role": "user", "content": f"{pregunta}\n\nContexto: {texto_preprocesado}"}
            ],
            temperature=temperatura
        )
        respuesta = response.choices[0].message['content'].strip()

        # Convertir la respuesta a audio
        tts = gTTS(text=respuesta, lang='es')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        # Mostrar la respuesta en texto y audio
        st.audio(audio_bytes, format="audio/mp3")
        return respuesta

    except openai.OpenAIError as e:
        st.error(f"Error al comunicarse con OpenAI: {e}")
        return "Lo siento, no puedo procesar tu solicitud en este momento."

def main():
    # --- Diseño general ---
    st.set_page_config(page_title="Asistente Virtual", page_icon="🤖")

    # --- Barra lateral ---
    with st.sidebar:
        st.image("logo omardent.png")
        st.title("🤖 Asistente Virtual BOTIDINAMIX AI")
        st.markdown("---")

        # Selección de modelo de lenguaje
        st.subheader("🧠 Configuración del Modelo")
        modelo = st.selectbox(
            "Selecciona el modelo:",
             ["gpt-3.5-turbo", "gpt-4","gpt-4o","gpt-4-32k","gpt-4-32k","gpt-3.5-turbo-16k,"],
            index=0,
            help="Elige el modelo de lenguaje de OpenAI que prefieras."
        )

        # --- Opciones adicionales ---
        st.markdown("---")
        temperatura = st.slider("🌡️ Temperatura", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # --- Video de fondo ---
    with st.container():
        st.markdown(
            f"""
            <style>
            #video-container {{
                position: relative;
                width: 100%;
                padding-bottom: 56.25%; /* Relación de aspecto 16:9 */
                background-color: lightblue; /* Fondo azul claro */
                overflow: hidden;  /* Evita que el video se desborde */
            }}
            #background-video {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
            </style>
            <div id="video-container">
                <video id="background-video" autoplay loop muted playsinline>
                    <source src="https://cdn.leonardo.ai/users/645c3d5c-ca1b-4ce8-aefa-a091494e0d09/generations/89dda365-bf17-4867-87d4-bd918d4a2818/89dda365-bf17-4867-87d4-bd918d4a2818.mp4" type="video/mp4">
                </video>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Área principal de la aplicación ---
    st.header("💬 Hablar con Galatea OMARDENT")

    # Carga de archivo PDF
    archivo_pdf = st.file_uploader("📂 Cargar PDF", type='pdf')

    # --- Chatbot ---
    if 'mensajes' not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta_usuario = st.chat_input("Pregunta:")
    if pregunta_usuario:
        st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        if archivo_pdf:
            texto_pdf = extraer_texto_pdf(archivo_pdf)
            texto_preprocesado = preprocesar_texto(texto_pdf)
        else:
            texto_preprocesado = ""  # Sin contexto de PDF si no se carga un archivo

        respuesta = obtener_respuesta(pregunta_usuario, texto_preprocesado, modelo, temperatura)
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)

if __name__ == "__main__":
    main()
