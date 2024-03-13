let blobs = [];
let stream;
let rec;
let recordUrl;
let audioResponseHandler;

const chatArea = document.getElementById('chat-area');
const messageInput = document.getElementById('messageInput');
const sendButton = document.querySelector('.msger-send-btn');

sendButton.addEventListener('click', enviarMensaje);

function enviarMensaje(event) {
  event.preventDefault();
  const messageText = messageInput.value.trim();
  if (!messageText) return;
  const userMessageElement = createMessageElement('Tú', messageText);
  chatArea.appendChild(userMessageElement);
  messageInput.value = '';
  enviarTexto(messageText);
}
// crea los mensajes en formato html que se muestran en el area de mensajes
function createMessageElement(name, text) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('msg');
  if (name === 'Tú') {
    messageElement.classList.add('right-msg');
  }

  const currentTime = new Date();
  const hours = currentTime.getHours().toString().padStart(2, '0'); // Agregar cero al principio si es necesario
  const minutes = currentTime.getMinutes().toString().padStart(2, '0');
  const seconds = currentTime.getSeconds().toString().padStart(2, '0');
  const formattedTime = `${hours}:${minutes}:${seconds}`;
  messageElement.innerHTML = `
    <div class="msg-bubble">
      <div class="msg-info">
        <div class="msg-info-name">${name}</div>
        <div class="msg-info-time">${formattedTime}</div>
      </div>
      <div class="msg-text">${text}</div>
    </div>
  `;
  return messageElement;
}

function enviarTexto(texto) {
  // Crear un objeto con los datos a enviar al servidor
  const data = { message: texto };

  // Generar una cadena de consulta única para evitar el caché
  const queryString = new URLSearchParams(data).toString();

  // Realizar una solicitud POST al servidor con la cadena de consulta única
  fetch(`/texto?${queryString}`, {  // Agregar la cadena de consulta única a la URL
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      // Verificar si la solicitud fue exitosa
      if (!response.ok) {
        throw new Error('Error al enviar el mensaje');
      }
      // Limpiar el campo de entrada de mensaje después de enviarlo
      messageInput.value = '';
      // Devolver la respuesta como JSON
      return response.json();
    })
    .then(data => {
      // Manejar la respuesta del servidor si es necesario
      //console.log('Respuesta del servidor:', data)
      const botMessage = createMessageElement('Asistente', data.respuesta);
      chatArea.appendChild(botMessage);
      chatArea.scrollTop = chatArea.scrollHeight; // Desplazar hacia abajo después de agregar el mensaje del asistente

      const checkbox = document.getElementById("vozCheckbox");
      // Verificar si está marcado para reproducir la voz  
      if (checkbox.checked) {
        // Agregar un identificador único al texto para enviar a enviarTextoVoz
        //const textoUnico = data.respuesta;
        //enviarTextoVoz(textoUnico);
        enviarTextoVoz(data.respuesta)
        //console.log(data.respuesta);
      } else {
        console.log("El checkbox no está marcado");
      }
    })
    .catch(error => {
      // Manejar errores de la solicitud
      console.error('Error:', error);
    });
}

function reproducirAudio(url) {
  fetch(url)
    .then(response => response.arrayBuffer())
    .then(buffer => {
      // Decodificar el ArrayBuffer en un audio
      return new AudioContext().decodeAudioData(buffer);
    })
    .then(audioBuffer => {
      // Crear un nuevo buffer de audio
      const audioContext = new AudioContext()
      const audioSource = new AudioBufferSourceNode(audioContext, { buffer: audioBuffer });
      // Conectar el buffer de audio al destino de salida (los altavoces)
      audioSource.connect(audioContext.destination);
      // Iniciar la reproducción
      audioSource.start();
    })
    .catch(error => {
      console.error('Error al reproducir el audio:', error);
    });
}

function enviarTextoVoz(texto) {
  const data = { message: texto };
  fetch('/texto_voz', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-cache'
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) {
        console.log('Error al enviar el mensaje');
      } else {
        return response.json();
      }
    })
    .then(data => {
      if (data && data.respuesta) {
        reproducirAudio("static/" + data.respuesta);
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
}


function recorder(url, handler) {
  recordUrl = url;
  if (typeof handler !== "undefined") {
    audioResponseHandler = handler;
  }
}

async function record() {
  try {
    document.getElementById("record1").style.display = "none";
    document.getElementById("stop1").style.display = "";
    document.getElementById("record-stop-label").style.display = "block"
    document.getElementById("record-stop-loading").style.display = "none"
    document.getElementById("stop1").disabled = false
    blobs = [];
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    rec = new MediaRecorder(stream);
    rec.ondataavailable = e => {
      if (e.data) {
        blobs.push(e.data);
      }
    }
    rec.onstop = doPreview;
    rec.start();
  } catch (e) {
    console.log(e)
    alert("No fue posible iniciar el grabador de audio! Favor de verificar que se tenga el permiso adecuado, estar en HTTPS, etc...");
  }
}

function doPreview() {
  if (!blobs.length) {
    // console.log("No hay blobios!");
  } else {
    //console.log("Tenemos blobios!");
    const blob = new Blob(blobs);
    var fd = new FormData();
    fd.append("audio", blob, "audio");
    fetch(recordUrl, {
      method: "POST",
      body: fd,
    })
      .then((response) => response.json())
      .then(audioResponseHandler)
      .catch(err => {
        console.log("Oops: Ocurrió un error al convertir la respuesta a JSON", err);
      })
      .then(() => {
        console.log("La promesa anterior se completó correctamente");
      })
      .catch(err => {
        console.log("Oops: Ocurrió un error en alguna parte de la cadena", err);
      });
  }
}

function stop() {
  //if (!recordingInProgress) return; // Check if recording is not in progress
  //recordingInProgress = false; // Reset recording in progress flag
  document.getElementById("record-stop-label").style.display = "none";
  document.getElementById("record-stop-loading").style.display = "block";
  document.getElementById("stop1").disabled = true;
  rec.stop();
}

window.onload = () => {
  recorder("/audio", response => {
    document.getElementById("record1").style.display = "";
    document.getElementById("stop1").style.display = "none";
    if (!response || response == null) {
      console.log("No response");
      return;
    }
    document.getElementById("messageInput").value = response.text;
    document.getElementById("enviar").click();
  });
};
