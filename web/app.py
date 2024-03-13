import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pathlib import Path
from gtts import gTTS
import re
from html import escape
#from youtube_transcript_api import YouTubeTranscriptApi
import json

# Cargar llaves del archivo .env
load_dotenv('.env')
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_key = os.environ.get("PINECONE_API_KEY")
model_name = 'text-embedding-ada-002'
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

indice = "videos"
embedding = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
client = OpenAI()

# Cargo el los vectores desde Pinecone
vector_store = Pinecone.from_existing_index(indice, embedding)

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Preparo la LLM
llm1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
crc = ConversationalRetrievalChain.from_llm(llm=llm1, retriever=retriever)
memoria = []

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

#Maneja las solicitudes POST  
@app.route("/texto", methods=["POST"])

def recibir_texto():  #recibe una pregunta en formato JSON
   mensaje = request.json.get("message")
   respuesta = texto(mensaje)
   return jsonify({"respuesta": respuesta})

def texto(pregunta): # busca la respuesta en la llm , convierte los enlaces si tiene para html y mailito
    respuesta = crc({"question": pregunta, "chat_history": memoria})
    memoria.append((pregunta, respuesta["answer"]))
    respuesta = convertir_urls_a_enlaces(respuesta["answer"])
    return respuesta

#Maneja las solicitudes POST para transcribir un archivo de audio recibido y devolver el texto transcrito.
@app.route("/audio", methods=["POST"])
def audio():
    audio = request.files.get("audio")
    audio.save("static/pregunta.mp3")
    audio_file = open("static/pregunta.mp3", "rb")
    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return {"result": "ok", "text": transcription.text}

@app.route("/texto_voz", methods=["POST"])#Maneja las solicitudes POST para recibir un mensaje en formato JSON y generar una respuesta de voz.
  
def recibir_texto_voz():
    mensaje = request.json.get("message")
    # Eliminar URL del mensaje
    patron_url = r'(https?://\S+)'
    mensaje = re.sub(patron_url, ' ', mensaje)
    # Eliminar dirección de correo electrónico del mensaje
    patron_email = r'[\w\-]+@[\w\.-]+'
    mensaje = re.sub(patron_email, ' ', mensaje)

    speech_file_path = Path(__file__).parent / "static/respuestaia.mp3"
    resp = client.audio.speech.create(model="tts-1", voice="alloy", input=mensaje)
    resp.stream_to_file(speech_file_path)
    return jsonify({"respuesta": "respuestaia.mp3"})



def convertir_urls_a_enlaces(texto):
    # Expresión regular para identificar URLs
    patron_url = r'(https?://\S+)'
    urls = re.findall(patron_url, texto)
    #minuto_inicio(urls)
    minuto_inicio=5.91
    # Reemplazar URLs con enlaces HTML
    for url in urls:
        #enlace_html = f'<a href="{escape(url)}" target="_blank">{escape(url)}</a>'
        #texto = texto.replace(url, enlace_html)
        if 'youtube.com' in url or 'youtu.be' in url:
            if '?' in url:
                url += f'&start={minuto_inicio * 60}'
            else:
                url += f'?start={minuto_inicio * 60}'
        
        enlace_html = f'<a href="{escape(url)}" target="_blank">{escape(url)}</a>'
        texto = texto.replace(url, enlace_html)

    # Expresión regular para identificar direcciones de correo electrónico
    patron_email = r'([\w\-]+@[\w\.-]+)'
    emails = re.findall(patron_email, texto)

    # Reemplazar direcciones de correo electrónico con enlaces de correo electrónico
    for email in emails:
        if email.endswith('.'):
            email = email[:-1]  # Eliminar el punto final si existe por que me lo entregaba con un . al final
        enlace_email = f'<a href="mailto:{email}">{email}</a>'
        texto = texto.replace(email, enlace_email)

    return texto


if __name__ == "__main__":
    app.run(debug=False)
