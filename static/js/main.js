var socket = null;
var tokenElement = null;
var outputElement = null;
var currentAudio = null;
var thinkerElement = null;

var md = markdownit();
var speechAccumulator = "";
var textAccumulator = "";
var utteranceQueue = [];
var micEmoji = 'Listen \u{1F399}';
var earEmoji = 'Listening... \u{1F442}';

async function getClipboardText() {

    const clipboardContents = await navigator.clipboard.read();
    for (const item of clipboardContents) {
        if (item.types.includes("image/png")) {
            dialogueElement = document.getElementById('dialogue');
            const blob = await item.getType("image/png");
            objectURL = URL.createObjectURL(blob);
            img = document.createElement("img");
            img.setAttribute("style", "width: 10em");
            img.setAttribute("src", objectURL);
            chatElement = document.createElement('div');
            chatElement.classList.add('chat');
            chatElement.classList.add('chat-end');
            //userTextElement.classList.add('user');
            img.classList.add('animate__animated');
            img.classList.add('animate__fadeInUp');
            img.classList.add('animate__delay-2s');
            img.classList.add('chat-bubble');
            img.classList.add('chat-bubble-accent');
            chatElement.append(img);
            dialogueElement.append(chatElement);
            var url = new URL('../../llava/describe', window.location);
            const response = await fetch(url, {
                method: "POST",
                headers: {
                  "Content-Type": "image/png",
                },
                body: blob,
              });
            desc = await response.text();
            img.setAttribute("alt", desc);
            return desc;
        } else if (item.types.includes("text/html")) {
            const blob = await item.getType("text/html");
            const blobText = await blob.text();
            var turndownService = new TurndownService()
            return turndownService.turndown(blobText);
        } else if (item.types.includes("text/plain")) {
            const blob = await item.getType("text/plain");
            return await blob.text();
        } else  {
            throw new Error("Clipboard does not contain textual data.");
        }
    }

}


/// Figure out the relative address to the webscket port for the given named personality.
makeSocketAddress = function (personality) {
    protocol = (window.location.protocol == "http:" ? "ws" : "wss");
    return protocol + "://" + window.location.host + "/gpt-socket/" + personality;
}

addUtterance = function () {
    if (speechAccumulator) {
        var personality = window.personality ? window.personality : "whisper";
        var encodedText = encodeURIComponent(speechAccumulator);
        if (document.getElementById("speech-toggle").checked) {
            var url = new URL('/tts/'+personality, window.location);
            url.search = new URLSearchParams({ text: speechAccumulator })
            currentAudio = new Audio(url);
            utteranceQueue.push(currentAudio);
            speechAccumulator = "";
            if (utteranceQueue.length == 1) {
                nextUtterance();
            }
        }
    }
}

nextUtterance = function () {
    if (utteranceQueue.length > 0) {
        utteranceQueue[0].onended = (event) => {
            utteranceQueue.shift();
            nextUtterance();
        }
        if (utteranceQueue[0].readyState >= 3) {
            utteranceQueue[0].play();
        } else {
            utteranceQueue[0].addEventListener('loadeddata', (e) => {
                utteranceQueue[0].play();
            });
        }
    }
}

loadEmotionalAffect = async function () {
    var personality = window.personality ? window.personality : "whisper";
    var url = new URL('../../toil/' + personality, window.location);
    var payload = { session: window.llmSessionId, prompt: "What is an appropriate 2-word emotional response to this conversation? Use a gerund verb and an adverb. Respond in JSON format, with a short explanation in the 'explanation' field",
        schema: {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {
                "description": {
                    "type": "string"
                  },
                "gerundVerb": {
                    "type": "string"
                  },
                  "adverb": {
                    "type": "string"
                  }
            },
            "required": [
                "gerundVerb", 
                "adverb",
                "description"
            ]
          }
     };
    fetch(url, {
        method: "POST",
        headers: {'Content-Type' : 'application/json'},
        body: JSON.stringify(payload) 
    }).then((resp) => resp.json()).then((json) => renderEmotionalAffect(window.persona + ", "+ json.gerundVerb +" " + json.adverb + " at the camera."));
}

renderEmotionalAffect = function (affect) {
    var url = new URL('../../stablediffusion/generate', window.location);
    url.search = new URLSearchParams({ 
        seed: window.persona_seed,
        cfg: window.persona_cfg,
        steps: window.persona_steps, 
        prompt: affect, 
        format:"JPEG" }).toString();

    window.lastBotImage.src = url;
}

needsEmotionalAffect = function () {
    var control = document.getElementById("emotional-affect");
    return (control && control.checked);
}

createWebSocket = function (prompt) {
    var personality = window.personality ? window.personality : "whisper";
    var ws = new WebSocket(makeSocketAddress(personality));
    var conversationStarter = prompt;
    ws.onopen = function () {
        ws.send(prompt);
    };
    ws.onclose = function (evt) {
        //        contextElement.hidden = false;
    };
    ws.onerror = function (evt) {
        console.log(evt)
    };
    ws.onmessage = function (evt) {
        token = evt.data;
        if (token.startsWith("<END ")) {
            addUtterance();
            window.llmSessionId = token.substring(5, token.length - 1)
            ws.close();
            if(needsEmotionalAffect())
            loadEmotionalAffect();
            // Do any UI work that marks the end of the interaction here.
        } else {
            textAccumulator += token;
            if (document.getElementById("speech-toggle").checked) {
                speechAccumulator = speechAccumulator.concat(evt.data);
            }
            if (evt.data.match(/[\.\!\?\:]/g)) {
                addUtterance();
            }
        }
        if (thinkerElement.parentElement) {
            thinkerElement.parentElement.removeChild(thinkerElement);
        }
        outputElement.innerHTML = md.render(textAccumulator);

        outputElement.scrollIntoView(false);
    };
    return ws;
}
showError = function (error) {
    dialogueElement = document.getElementById('dialogue');
    var errorTextElement = document.createElement('div');
    errorElement = document.createElement('div');
    errorElement.classList.add('chat');
    //userTextElement.classList.add('user');
    errorTextElement.classList.add('animate__animated');
    errorTextElement.classList.add('animate__fadeInUp');
    errorTextElement.textContent = error;

    errorTextElement.classList.add('chat-bubble');
    errorTextElement.classList.add('chat-bubble-error');
    errorElement.append(errorTextElement);
    dialogueElement.append(errorElement);
}
sendPrompt = async function () {
    contextElement.hidden = true;
    var promptElement = document.getElementById('prompt');
    newBotImage = document.getElementById("botImage").cloneNode(true);
    window.lastBotImage = newBotImage;
    dialogueElement = document.getElementById('dialogue');
    var userTextElement = document.createElement('div');
    chatElement = document.createElement('div');
    chatElement.classList.add('chat');
    chatElement.classList.add('chat-end');

    //userTextElement.classList.add('user');
    userTextElement.classList.add('animate__animated');
    userTextElement.classList.add('animate__fadeInUp');
    userTextElement.textContent = promptElement.value;

    userTextElement.classList.add('chat-bubble');
    userTextElement.classList.add('chat-bubble-accent');
    chatElement.append(userTextElement);
    dialogueElement.append(chatElement);

    chatElement.classList.add('animate__animated');
    chatElement.classList.add('animate__fadeIn');
    chatElement.classList.add('animate__delay-1s');

    if (socket && socket.readyState == 1) {
        socket.send(promptElement.value);
    } else {
        var nextMessage = promptElement.value;
        if (window.llmSessionId) {
            nextMessage = "|SESSION|" + window.llmSessionId + "|/SESSION|" + promptElement.value;
        } else if (window.firstContextText) {
            nextMessage = "|CONTEXT|" + window.firstContextText + "|/CONTEXT|" + promptElement.value;
        } else if (document.getElementById('clipboard-rag-checkbox') && document.getElementById('clipboard-rag-checkbox').checked) {
            ragText = await getClipboardText();
            nextMessage = "|CONTENT|" + ragText + "|/CONTENT|" + promptElement.value;
        } else if (document.getElementById('wikipedia-rag-checkbox') && document.getElementById('wikipedia-rag-checkbox').checked) {
            nextMessage = "|RAG|wikipedia.org|/RAG|" + promptElement.value;
        } else if (document.getElementById('goc-rag-checkbox') && document.getElementById('goc-rag-checkbox').checked) {
            nextMessage = "|RAG|canada.ca|gc.ca|/RAG|" + promptElement.value;
        }
        socket = createWebSocket(nextMessage)
    }

    chatElement = document.createElement('div');
    chatElement.classList.add('chat');
    chatElement.classList.add('chat-start');
    dialogueElement.append(chatElement);
    avatarHolderElement = document.createElement('div');
    avatarHolderElement.classList.add(needsEmotionalAffect() ? "w-32" : "w-10");
    avatarHolderElement.classList.add("rounded-full");
    avatarHolderElement.append(newBotImage);
    avatarElement = document.createElement('div');
    avatarElement.classList.add('chat-image');
    avatarElement.classList.add('avatar');
    avatarElement.append(avatarHolderElement);
    chatElement.append(avatarElement);
    outputElement = document.createElement('div');
    outputElement.classList.add('chat-bubble');
    outputElement.classList.add('chat-bubble-primary');
    outputElement.classList.add('prose');
    chatElement.append(outputElement);
    chatElement.classList.add('animate__animated');
    chatElement.classList.add('animate__fadeIn');
    chatElement.classList.add('animate__delay-2s');
    outputElement.append(thinkerElement);
    
    promptElement.value = '';
    textAccumulator = "";
    userTextElement.scrollIntoView(false);
}
startDictation = function () {
    listenButton = document.getElementById('listen-button');
    if (window.hasOwnProperty('webkitSpeechRecognition')) {
        var recognition = new webkitSpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        listenButton.innerText = earEmoji;
        recognition.start();

        recognition.onresult = function (e) {
            document.getElementById('prompt').value = e.results[0][0].transcript;
            recognition.stop();
            listenButton.innerText = micEmoji;
            sendPrompt();
        };
        recognition.onerror = function (e) {
            recognition.stop();
        };
    }
}

showModelInformation = async function () {
    const response = await fetch("../describe/" + window.personality);
    const jsonData = await response.json();

    // Create table
    var table = document.createElement("table");
    table.classList.add("table");
    table.classList.add("w-full");

    // Create table body
    var tbody = document.createElement("tbody");

    // Add data
    for (var key in jsonData) {
        var tr = document.createElement("tr");

        var tdKey = document.createElement("th");
        tdKey.textContent = key;
        tr.appendChild(tdKey);

        var tdValue = document.createElement("td");
        tdValue.textContent = jsonData[key];
        tr.appendChild(tdValue);

        tbody.appendChild(tr);
    }

    document.getElementById('descriptionName').textContent = window.personality;
    document.getElementById('description').textContent = '';
    document.getElementById('description').appendChild(table);
    table.appendChild(tbody);


    document.getElementById("descriptionDialog").showModal();

}
//socket = createWebSocket();
