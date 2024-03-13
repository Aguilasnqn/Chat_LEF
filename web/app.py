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
llm1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5,max_tokens=1024)
#llm1 = ChatOpenAI(model="text-davinci-003", temperature=0.7,max_tokens=1024)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
crc = ConversationalRetrievalChain.from_llm(llm=llm1, retriever=retriever)
memoria = []
ubicartiempo=YouTubeTranscriptApi

# Resumir usa estas librerias
#from langchain import PromptTemplate     no funciona mas
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
#from langchain.document_loaders import YoutubeLoader no funciona mas
from langchain_community.document_loaders import YoutubeLoader
#from langchain.chat_models import ChatOpenAI no funciona mas
#from langchain_community.chat_models import ChatOpenAI no funciona mas
from langchain.text_splitter import RecursiveCharacterTextSplitter as RC 
from langchain.chains.summarize import load_summarize_chain
#from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader
import pyperclip

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

def texto(pregunta): # busca la respuesta en la llm , convierte los enlaces si tiene para html y mailito, tambien detecta si hay un resumen de algun video
#    respuesta = crc({"question": pregunta, "chat_history": memoria})
     # identifica si quiere hacer un resumen
        patron_palabras = r'\b(resumen)\b'
        coincidencias = re.findall(patron_palabras, pregunta, flags=re.IGNORECASE)
        if coincidencias: # hace el resumen del link
            patron_url = r'(https?://\S+(?:&\S+)*)'
            url = re.findall(patron_url, pregunta)
            if url:
               respuesta = resumir(url[0])  # Solo se toma la primera URL si hay múltiples
               return respuesta
        respuesta = crc({"question": pregunta,"chat_history": memoria})

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
    # hay que extrer el link generado por que si no da un error
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
        
        
# Imprimir las coincidencias encontradas
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
        texto_transcripcion=""
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



def resumir(url):
    #url='https://www.youtube.com/watch?v=XLAdvjbQe6M'
    # Cargar transcripción del video de YouTube
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=["es"])
    transcripcion = loader.load()

    # Dividir la transcripción en fragmentos
    text_splitter = RC(chunk_size=2000, chunk_overlap=20)
    fragmentos = text_splitter.split_documents(transcripcion)

    # Inicializar el modelo de inteligencia artificial
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)

    # Definir plantillas de texto para las solicitudes iniciales y de refinamiento del resumen
    prompt_template = """Escribe un resumen de lo siguiente extrayendo la información clave:
    Text: {text}
    Resumen:"""
    prompt_inicial = PromptTemplate(template=prompt_template, input_variables=['text'])

    refine_template = '''
    Tu trabajo consiste en elaborar un resumen final detallado.
    He proporcionado un resumen existente hasta cierto punto: {existing_answer}.
    Por favor, perfeccione el resumen existente con algo más de contexto a continuación.
    ------------
    {text}
    ------------
    Comience el resumen final con una "Introducción" que ofrezca una visión general del tema seguido
    por los puntos más importantes ("Bullet Ponts"). Termina el resumen con una conclusión.
    '''
    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_answer', 'text']
    )

    # Cargar y ejecutar la cadena de resumen
    chain = load_summarize_chain(llm=llm, chain_type='refine', question_prompt=prompt_inicial, refine_prompt=refine_prompt, return_intermediate_steps=False)
    resumen = chain.run(fragmentos)
    resumen_formateado = f"Resumen:\n\n{resumen}"
    pyperclip.copy(resumen_formateado)
   # resumen_formateado = "Resumen:\n\n" + "\n\n".join(resumen)
    resumen_html = f"<textarea rows='20' cols='90'>{resumen_formateado}</textarea>"

    return resumen_html




if __name__ == "__main__":
    app.run(debug=True)