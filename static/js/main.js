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

/// Figure out the relative address to the webscket port for the given named personality.
makeSocketAddress = function (personality) {
    protocol = (window.location.protocol == "http:" ? "ws" : "wss");
    return protocol + "://" + window.location.host + "/gpt-socket/" + personality;
}

addUtterance = function () {
    if (speechAccumulator) {
        var encodedText = encodeURIComponent(speechAccumulator);
        if (document.getElementById("speech-toggle").checked) {
            // alternate voice: currentAudio = new Audio("https://psx-xsp.csps-efpc.ca/tts?&voice=en_US%2Fvctk_low%23p283&noiseScale=0.667&noiseW=0.8&lengthScale=1&ssml=false&audioTarget=client&text=" + encodedText);
            if (document.documentElement.getAttribute("lang") == "fr") {
                currentAudio = new Audio("https://psx-xsp.csps-efpc.ca/tts?&voice=fr_FR%2Fm-ailabs_low%23nadine_eckert_boulet&noiseScale=0.667&noiseW=0.8&lengthScale=1&ssml=false&audioTarget=client&text=" + encodedText);
            } else {
                currentAudio = new Audio("https://psx-xsp.csps-efpc.ca/tts?&voice=en_US%2Fhifi-tts_low%2392&noiseScale=0.667&noiseW=0.8&lengthScale=1&ssml=false&audioTarget=client&text=" + encodedText);
            }
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
    ws.onmessage = function (evt) {
        token = evt.data;
        if (token.startsWith("<END ")) {
            addUtterance();
            window.llmSessionId = token.substring(5, token.length - 1)
            ws.close();
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
sendPrompt = function () {
    contextElement.hidden = true;
    var promptElement = document.getElementById('prompt');
    newBotImage = document.getElementById("botImage").cloneNode(true);

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



    chatElement = document.createElement('div');
    chatElement.classList.add('chat');
    chatElement.classList.add('chat-start');
    dialogueElement.append(chatElement);
    avatarHolderElement = document.createElement('div');
    avatarHolderElement.classList.add("w-10");
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
    if (socket && socket.readyState == 1) {
        socket.send(promptElement.value);
    } else {
        var nextMessage = promptElement.value;
        if (window.llmSessionId) {
            nextMessage = "|SESSION|" + window.llmSessionId + "|/SESSION|" + promptElement.value;
        } else if (window.firstContextText) {
            nextMessage = "|CONTEXT|" + window.firstContextText + "|/CONTEXT|" + promptElement.value;
        } else if (document.getElementById('wikipedia-rag-checkbox').checked) {
            nextMessage = "|RAG|wikipedia.org|/RAG|" + promptElement.value;
        } else if (document.getElementById('goc-rag-checkbox').checked) {
            nextMessage = "|RAG|canada.ca|gc.ca|/RAG|" + promptElement.value;
        }
        socket = createWebSocket(nextMessage)
    }
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
introSpeech = function (firstPhrase) {
    var dialogueElement = document.getElementById('dialogue');
    chatElement = document.createElement('div');
    chatElement.classList.add('chat');
    chatElement.classList.add('chat-start');
    avatarElement = document.createElement('div');
    avatarElement.classList.add('chat-image');
    avatarElement.classList.add('avatar');
    avatarHolderElement = document.createElement('div');
    avatarHolderElement.classList.add("w-10");
    avatarHolderElement.classList.add("rounded-full");
    avatarHolderElement.append(botImage);
    avatarElement.append(avatarHolderElement);
    chatElement.append(avatarElement);
    outputElement = document.createElement('div');
    outputElement.classList.add('chat-bubble');
    outputElement.classList.add('chat-bubble-primary');

    outputElement.append(document.createTextNode(firstPhrase));
    // outputElement.classList.add('bot');
    dialogueElement.append(chatElement);
    chatElement.append(outputElement);
    outputElement.classList.add('animate__animated');
    outputElement.classList.add('animate__fadeIn');
    outputElement.classList.add('animate__delay-2s');
}


//socket = createWebSocket();
