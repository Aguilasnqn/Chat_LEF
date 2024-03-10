import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify # Servidor
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pathlib import Path
from gtts import gTTS
import re
from html import escape
from youtube_transcript_api import YouTubeTranscriptApi # Busco la linea de tiempo de un video en particular
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
ubicartiempo=YouTubeTranscriptApi

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
    
    respuesta = convertir_urls_a_enlaces(respuesta["answer"],pregunta)
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
    # Obtener el texto del mensaje enviado desde el frontend
    mensaje = request.json.get("message")
    patron_url = r'(https?://\S+)'
    
    # Reemplazar todas las URLs en la cadena de texto por el texto "Click en el Link"
    # hay que extrer el lik generado por que si no da un error
    mensaje = re.sub(patron_url, 'Click en el Link', mensaje)



    speech_file_path = Path(__file__).parent / "static/respuestaia.mp3"
    resp = client.audio.speech.create(
           model="tts-1",
           voice="alloy",
           input=mensaje
    )

    resp.stream_to_file(speech_file_path)
    


    # Guardar el archivo de audio en el directorio "static"
    #with open(speech_file_path , "wb") as f:
    #     f.write(resp.audio_content)
    #tts=gTTS(mensaje,lang="es",tld="com.mx") 
    #tts.save("peech_file_path")

    # Procesar el mensaje utilizando la función texto

    # Devolver la respuesta al frontend
    return jsonify({"respuesta":"respuestaia.mp3"})
  



def convertir_urls_a_enlaces(texto, pregunta1):
    # Expresión regular para identificar URLs
    patron_url = r'(https?://\S+(?:&\S+)*)'
    urls = re.findall(patron_url, texto)
    if urls: # Si hay URL, procesar
        palabra_clave = obtener_palabra_clave(pregunta1)
        enlaces_modificados = {}
        for url in urls:
            minuto_comienzo = buscar_minuto_comienzo(palabra_clave, url)
            if minuto_comienzo is not None:
                if 'youtube.com' in url or 'youtu.be' in url:
                    segundos_inicio = int(minuto_comienzo * 60)
                    if '?' in url:
                        url_con_tiempo = f'{url}&t={segundos_inicio}s'
                    else:
                        url_con_tiempo = f'{url}?t={segundos_inicio}s'
                    enlaces_modificados[url] = f'<a href="{escape(url_con_tiempo)}" target="_blank">{escape(url_con_tiempo)}</a>'
                   # enlaces_modificados[url] = url_con_tiempo
                        
        # Reemplazar cada enlace en el texto con su versión modificada
      
        for url, enlace_html in enlaces_modificados.items():
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


def buscar_minuto_comienzo(clave, url1):
    patron = r"(?<=v=)([\w-]+)"
    id = re.search(patron, url1)
    if id:
        _id = id.group(0)
        transcripcion = ubicartiempo.get_transcript(_id, languages=('es',))
        if transcripcion:
            for item in transcripcion:
                if 'text' in item and 'start' in item:
                    if clave in item['text']:
                        return item['start'] / 60  # Convertir segundos a minutos
    return 0.66 # Devolver 40 segundos es el tiempo de la presentacion si no se encuentra la clave


def obtener_palabra_clave(frase):
    patron_palabra_clave = r'(?:clase|video|contexto|link|enlace)\s+(?:número\s+\d+\s+)?(?:de\s+)?(\w+)'
    # Buscar la palabra clave en la frase
    coincidencia = re.search(patron_palabra_clave, frase, re.IGNORECASE)
    if coincidencia:
        return coincidencia.group(1)  # Devolver la palabra clave encontrada
    else:
        return ""

if __name__ == "__main__":
    app.run(debug=True)