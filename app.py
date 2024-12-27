# Import necessary libraries
import gc
import re
import os
import threading
import traceback
import subprocess
import uuid
import json
import random
import time
import io
import base64
import csv
from datetime import datetime
import unidecode
from duckduckgo_search import DDGS
import wikipedia
from simple_websocket import Server
from llama_cpp import Llama
from llama_cpp.llama_chat_format import NanoLlavaChatHandler
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from stable_diffusion_cpp import StableDiffusion
import stable_diffusion_cpp as sd_cpp
from PIL import Image
import flask
from flask import redirect, render_template, request, abort
import plotly.express as px

# Import nearby python files
import rag

# Feature Flags
SPECULATIVE_DECODING = int(os.getenv("WHISPER_SPECULATIVE_DECODING", "0"))
COMPRESS_AUDIO_TO_MP3 = os.path.exists("/usr/bin/lame")
EMBEDDING_MODEL = os.getenv(
    "WHISPER_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF"
)


# Initialize the Flask app and a thread lock for the LLM model
app = flask.Flask(__name__)  # , static_url_path=''
lock = threading.Lock()
tts_lock = threading.Lock()
# Lock for synchronizing log writing. Can be removed when we get async logging.
logLock = threading.Lock()


# Global settings and constants
stopTokens = [
    "<|assistant|>",
    "<|user|>",
    "<|end|>",
    "[/INST]",
    "[INST]",
    "</s>",
    "User:",
    "Assistant:",
    "[/ASK]",
    "[INFO]",
    "<</SYS>>",
]
temperature = 0.8
session_cache_size = 100
systemless_markers = [
    "granite-8b-code-instruct",
    "Phi-3-medium",
    "Mistral-7B-Instruct-v0.3",
    "OLMo-7B-Instruct",
    "gemma-2-",
]

# Prompt parts
system_prefix = "[INST]\n"
system_suffix = "\n[/INST]\n"
prompt_prefix = "\nUser: "
prompt_suffix = " \n"
response_prefix = "Assistant: "
response_suffix = ""
rag_prefix = "\nConsider the following:\n"
rag_suffix = "\nGiven the preceding text, "
pleaseWaitText = "\n[Please note that I'm currently helping another user and will be with you as soon as they've finished.]\n"

__cached_sd = None
__cached_sd_modelName = None
__cached_llm = None
__cached_personality = None
__cached_sessions = {}


def getSd(modelName):
    global __cached_sd
    global __cached_sd_modelName
    if not lock.locked():
        # The method has been called by a thread not holding the lock.
        raise Exception(
            "Attempted to control the loaded model without holding the exclusivity lock."
        )
    if __cached_sd_modelName == modelName:
        log_event(subject="cache", eventtype="cache_hit")
        print("Using existing SD model!")
        return __cached_sd
    free_models()
    # Instantiate the model
    __cached_sd = StableDiffusion(
        model_path="../" + modelName,
        vae_decode_only=True,
        vae_path="../sdxl.vae.safetensors",
    )
    __cached_sd_modelName = modelName
    return __cached_sd


# Function to get the LLM model based on the provided personality
def getLlm(personality):
    global __cached_llm
    global __cached_personality
    if not lock.locked():
        # The method has been called by a thread not holding the lock.
        raise Exception(
            "Attempted to control the loaded model without holding the exclusivity lock."
        )
    if personality == __cached_personality:
        log_event(subject="cache", eventtype="cache_hit")
        return __cached_llm
    free_models()
    draft_model = None
    if SPECULATIVE_DECODING > 0:
        draft_model = LlamaPromptLookupDecoding(num_pred_tokens=SPECULATIVE_DECODING)
    model_spec = rag.get_model_spec(personality)
    if model_spec["local_file"] is not None:
        __cached_llm = Llama(
            model_path=model_spec["local_file"],
            n_gpu_layers=model_spec["gpu_layers"],
            n_threads=model_spec["cpu_threads"],
            numa=False,
            verbose=True,
            n_ctx=model_spec["context_window"],
            flash_attn=(model_spec["flash_attention"] == "true"),
            draft_model=draft_model,
        )
    else:
        __cached_llm = Llama.from_pretrained(
            repo_id=model_spec["hf_repo"],
            filename=model_spec["hf_filename"],
            verbose=True,
            n_gpu_layers=model_spec["gpu_layers"],
            n_threads=model_spec["cpu_threads"],
            numa=False,
            n_ctx=model_spec["context_window"],
            flash_attn=(model_spec["flash_attention"] == "true"),
            draft_model=draft_model,
        )
    __cached_personality = personality
    return __cached_llm


# Function to get the embedding model from the given HF repo
def getEmbedding(embedder):
    global __cached_llm
    global __cached_personality
    if not lock.locked():
        # The method has been called by a thread not holding the lock.
        raise Exception(
            "Attempted to control the loaded model without holding the exclusivity lock."
        )
    if embedder == __cached_personality:
        log_event(subject="cache", eventtype="cache_hit")
        return __cached_llm
    free_models()
    __cached_llm = Llama.from_pretrained(
        repo_id=embedder, filename="*Q5_K_M.gguf", verbose=True, embedding=True
    )
    __cached_personality = embedder
    return __cached_llm


# Unload any cached models
def free_models():
    global __cached_llm
    global __cached_sd
    global __cached_sd_modelName
    global __cached_personality
    if not lock.locked():
        # The method has been called by a thread not holding the lock.
        raise Exception(
            "Attempted to control the loaded model without holding the exclusivity lock."
        )
    if __cached_llm:
        del __cached_llm
        gc.collect()
        __cached_llm = None
        __cached_personality = None
        time.sleep(0.5)
    if __cached_sd:
        del __cached_sd
        gc.collect()
        __cached_sd = None
        __cached_sd_modelName = None
        time.sleep(0.5)


def isSystemlessModel(repo_name):
    return any(map(lambda marker: marker in repo_name, systemless_markers))


# Flask route to bounce users to the default UI
@app.route("/")
def root_redir():
    return redirect("/chat", code=302)


# A service directory page
@app.route("/chat")
def chat_directory():
    return render_template(
        "chat_directory.html", personalities=rag.personalities, rag=rag
    )


# A rudimentary stats endpoint
@app.route("/platform/stats")
def render_stats():
    with logLock:
        gen_events = []
        times = []
        durations = []
        color_hints = []
        plot = ""
        epoch = datetime.fromtimestamp(0)
        with open("log.csv", mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[3] in [
                    "cache_hit",
                    "message_socket",
                    "start_gpt",
                    "end_socket",
                    "end_gpt",
                    "end_toil",
                    "end_generation",
                ]:
                    gen_events.append(
                        [
                            int(
                                (datetime.fromisoformat(row[0]) - epoch).total_seconds()
                                * 1000
                            ),
                            row[2],
                            row[3],
                            "miss",
                            row[5],
                        ]
                    )
        for index, row in enumerate(gen_events):
            if row[2] == "cache_hit":
                gen_events[index - 1][3] = "hit"
        cache_hits = sum(
            1
            for i in gen_events
            if i[2] in ["message_socket", "start_gpt"] and i[3] == "hit"
        )
        cache_misses = sum(
            1
            for i in gen_events
            if i[2] in ["message_socket", "start_gpt"] and i[3] == "miss"
        )
        for index, row in enumerate(gen_events):
            if row[2].startswith("end_"):
                times.append(row[0])
                durations.append(int(row[4]))
                color_hints.append(row[2])

        plot = px.scatter(x=times, y=durations, color=color_hints).to_html(
            include_plotlyjs="cdn"
        )

        return render_template(
            "stats.html", cache_hits=cache_hits, cache_misses=cache_misses, plot=plot
        )


# Flask route for handling websocket connections for user conversations
@app.route("/chat/<personality>")
def render_chat(personality):
    if rag.personality_exists(personality):
        print(rag.get_model_spec(personality)["ui_features"])
        return render_template(
            "chat.html",
            personality=personality,
            model_spec=rag.get_model_spec(personality),
            ui_features=rag.get_model_spec(personality)["ui_features"],
        )
    else:
        return abort(404)


# Flask route for handling websocket connections for user conversations
@app.route("/gpt-socket/<personality>", websocket=True)
def gpt_socket(personality):
    ws = Server.accept(request.environ)
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/gpt-socket/" + personality,
        eventtype="start_socket",
    )
    start = datetime.now()
    # We receive and parse the first user prompt.
    message = ws.receive()
    folded = message.casefold()
    url = None
    text = None
    rag_domain = None
    sessionkey = uuid.uuid4().urn
    rag_source_description = ""
    model_spec = rag.get_model_spec(personality)
    rag_personality = personality
    if model_spec["rag_helper"]:
        rag_personality = model_spec["rag_helper"]
    ## TODO: make this bit modular.
    if message.startswith("|SESSION|"):
        s = message[9:].split("""|/SESSION|""", 1)
        message = s[1]
        sessionkey = s[0]
    elif message.startswith("|CONTEXT|"):
        s = message[9:].split("""|/CONTEXT|""", 1)
        message = s[1]
        text = s[0]
        ws.send("Reading the provided context...")
    elif message.startswith("|CONTENT|"):
        s = message[9:].split("""|/CONTENT|""", 1)
        message = s[1]
        text = s[0]
        ws.send("Reading the provided content...")
    elif message.startswith("|RAG|"):
        s = message[5:].split("""|/RAG|""", 1)
        message = s[1]
        rag_domain = s[0]
        reflection = None
        matches = None
        with lock:
            reflection = ask(
                'If the following text is a question that could be answered with a web search, answer with a relevant search term. Answer with only the terms surrounded by " characters, or "none" if the question is inappropriate. Do not answer anything after the quoted string.\n\n'
                + message,
                personality=rag_personality,
            )
            print(reflection)
            matches = re.search(r'"([^"]+)"', reflection)
            if matches is not None and matches.group(1) != "none":
                if rag_domain == "wikipedia.org":
                    ws.send('Searching wikipedia for "' + matches.group(1) + '" ... ')
                    results = wikipedia.search(matches.group(1))
                    if results:
                        results = ask(
                            'Sort the following topics into a JSON array like ["first","second","third"] from highest relevance to lowest for the query "'
                            + matches.group(1)
                            + '":\n'
                            + "\n".join(results),
                            personality=rag_personality,
                            schema={"type": "array", "items": {"type": "string"}},
                        )
                        for i in range(3):
                            if i < 3 and text is None:
                                try:
                                    result = wikipedia.page(results[i])
                                    possible_text = result.content[
                                        : model_spec["rag_length"]
                                    ]
                                    ws.send(
                                        "Evaluating search result ["
                                        + result.title
                                        + "]("
                                        + result.url
                                        + ") ..."
                                    )
                                    ws.send("\n\n")
                                    ask_response = ask(
                                        "Consider the following content: \n\n"
                                        + possible_text
                                        + '\n\nAnswer "true" or "false": Is the content about "'
                                        + matches.group(1)
                                        + '"?',
                                        force_boolean=True,
                                        personality=rag_personality,
                                    )
                                    if ask_response:
                                        ws.send("Content is relevant:")
                                        ws.send("\n\n")
                                        rag_source_description = (
                                            'The page "'
                                            + result.title
                                            + '" at '
                                            + result.url
                                            + " says:\n"
                                        )
                                        text = possible_text
                                        print(
                                            'Accepting the page "'
                                            + result.title
                                            + '" at '
                                            + result.url
                                            + " as relevant."
                                        )
                                    else:
                                        print(
                                            'Discarding the page "'
                                            + result.title
                                            + '" at '
                                            + result.url
                                            + " as irrelevant."
                                        )
                                except wikipedia.exceptions.PageError:
                                    pass
                    else:
                        ws.send("Couldn't find a suitable reference.")
                        ws.send("\n\n")
                else:
                    ws.send('Searching for "' + matches.group(1) + '" ... ')
                    query = (
                        matches.group(1)
                        + " site:"
                        + (" OR site:".join(rag_domain.split("|")))
                    )
                    results = DDGS().text(query)
                    if results:
                        for i, result in enumerate(results, start=1):
                            if i < 3 and text is None:
                                possible_text = rag.fetch_url_text(
                                    results[i]["href"], model_spec["rag_length"]
                                )
                                ws.send("Evaluating search result " + str(i) + "...")
                                ws.send("\n\n")
                                if ask(
                                    "Consider the following content: \n\n"
                                    + possible_text
                                    + '\n\nIs the content about "'
                                    + matches.group(1)
                                    + '"?',
                                    force_boolean=True,
                                    personality=rag_personality,
                                ):
                                    ws.send(
                                        "Content is relevant: ["
                                        + results[i]["title"]
                                        + "]("
                                        + results[i]["href"]
                                        + "):"
                                    )
                                    ws.send("\n\n")
                                    rag_source_description = (
                                        'The page "'
                                        + results[i]["title"]
                                        + '" at '
                                        + results[i]["href"]
                                        + " says:\n"
                                    )
                                    text = possible_text
                                    print(
                                        'Accepting the page "'
                                        + results[i]["title"]
                                        + '" at '
                                        + results[i]["href"]
                                        + " as relevant."
                                    )
                                else:
                                    print(
                                        'Discarding the page "'
                                        + results[i]["title"]
                                        + '" at '
                                        + results[i]["href"]
                                        + " as irrelevant."
                                    )
                    else:
                        ws.send("Couldn't find a suitable reference.")
                        ws.send("\n\n")
            else:
                ws.send("No search necessary.")
                ws.send("\n\n")
    elif message.startswith("http"):
        s = message.split(" ", 1)
        message = s[1]
        url = s[0]
        ws.send("Fetching the page at " + url)
    elif "news".casefold() in folded:
        rag_source_description = "The latest news from the CBC is:\n"
        url = "https://www.cbc.ca/webfeed/rss/rss-topstories"
        if "canad".casefold() in folded:
            url = "https://www.cbc.ca/webfeed/rss/rss-canada"
        if "government".casefold() in folded:
            rag_source_description = (
                "The latest news from the Government of Canada is:\n"
            )
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?sort=publishedDate&orderBy=desc&pick=5&format=atom&atomtitle=National%20News"
        if "politi".casefold() in folded:
            url = "https://www.cbc.ca/webfeed/rss/rss-politics"
        if "health".casefold() in folded:
            rag_source_description = (
                "The latest health news from the Government of Canada is:\n"
            )
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?topic=health&sort=publishedDate&orderBy=desc&pick=5&format=atom&atomtitle=Health"
        if "tech".casefold() in folded:
            url = "https://www.cbc.ca/webfeed/rss/rss-technology"
        if "gaming".casefold() in folded:
            url = "https://kotaku.com/rss"
        if "sport".casefold() in folded:
            url = "https://www.cbc.ca/webfeed/rss/rss-sports"
        if "financ".casefold() in folded:
            url = "https://financialpost.com/feed"
        if "ottawa".casefold() in folded:
            url = "https://www.cbc.ca/webfeed/rss/rss-canada-ottawa"
        if "ottawa citizen".casefold() in folded:
            url = "https://ottawacitizen.com/feed"
        ws.send("Checking the news...\n")
    elif ("csps".casefold() in folded or "school".casefold() in folded) and (
        "cours".casefold() in folded or "learn".casefold() in folded
    ):
        rag_source_description = "The latest learning products from the CSPS are:\n"
        url = "https://www.csps-efpc.gc.ca/stayconnected/csps-rss-eng.xml"
        ws.send("Checking the catalogue...\n")
    elif (
        "open".casefold() in folded or "new".casefold() in folded
    ) and "dataset".casefold() in folded:
        rag_source_description = (
            "The latest open datasets from the Government of Canada are:\n"
        )
        url = "https://open.canada.ca/data/en/feeds/dataset.atom"
        ws.send("Checking the open data portal...\n")
    elif (
        "weather".casefold() in folded
        or "rain".casefold() in folded
        or "snow".casefold() in folded
        or "temperature".casefold() in folded
    ) and (
        "today".casefold() in folded
        or "this week".casefold() in folded
        or "tomorrow".casefold() in folded
        or "now".casefold() in folded
        or "later".casefold() in folded
        or "forecast".casefold() in folded
    ):
        rag_source_description = "The weather forecast for the Ottawa area is:\n"
        url = "https://weather.gc.ca/rss/city/on-118_e.xml"
        ws.send("Checking the forecast...\n")
    elif ("nouvelles").casefold() in folded:
        rag_source_description = "Les dernieres nouvelles de La Presse sont :\n"
        url = "https://www.lapresse.ca/actualites/rss"
        ws.send("Je rassemble les actualités...\n")

    chat_session = []

    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/gpt-socket/" + personality,
        eventtype="message_socket",
        data=message,
        session_id=sessionkey,
    )
    try:
        if sessionkey in __cached_sessions:
            chat_session = __cached_sessions.get(sessionkey, "")
            chat_session.append({"role": "user", "content": message})
        else:
            first_prompt = ""
            if (
                "agent_rag_source" in model_spec
                and model_spec["agent_rag_source"] is not None
            ):
                if os.path.isfile(model_spec["agent_rag_source"]):
                    with open(model_spec["agent_rag_source"]) as rag_file:
                        first_prompt += rag_file.read()
            if isSystemlessModel(model_spec["hf_repo"]):
                chat_session.append(
                    {"role": "user", "content": rag.get_personality_prefix(personality)}
                )
                chat_session.append({"role": "system", "content": ""})
            else:
                chat_session.append(
                    {
                        "role": "system",
                        "content": rag.get_personality_prefix(personality),
                    }
                )
            if url is not None:
                text = rag.fetch_url_text(url, model_spec["rag_length"])
            if text is not None:
                if(rag_source_description is not None) :
                    first_prompt += (
                        rag_source_description
                        + "\n"
                        + text
                        + "\nGiven the preceding content, "
                    )
                else:
                    first_prompt += (
                        "Consider the following content:\n"
                        + text
                        + "\nGiven the preceding content, "
                    )
            first_prompt += message
            chat_session.append({"role": "user", "content": first_prompt})

        if lock.locked():
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/gpt-socket/" + personality,
                eventtype="contention",
            )
        with lock:
            llm = getLlm(personality)

            stream = llm.create_chat_completion(
                chat_session,
                max_tokens=model_spec["context_window"],
                stop=stopTokens,
                stream=True,
                temperature=temperature,
            )
            response = ""
            for tok in stream:
                if "content" in tok["choices"][0]["delta"].keys():
                    token_string = tok["choices"][0]["delta"]["content"]
                    response += token_string
                    print(token_string, end="", flush=True)
                    ws.send(token_string)
            # For some reason - this method doesn't require the EOS token in the stream?!?! chat_session += llm.token_eos()
            ws.send("<END " + sessionkey + ">")
            ws.send(" ")  # Junk frame to ensure the previous one gets flushed?
            print("End session " + sessionkey)
            # We prepare the session for a subsequent user prompt, and the cycle begins anew.
            chat_session.append({"role": "assistant", "content": response})
            # TODO: if and when the cache moves out of process memory to a K-V store, the chat session will need to get written to that cache.
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/gpt-socket/" + personality,
                eventtype="end_socket",
                session_id=sessionkey,
                data=millis_since(start),
            )
    except Exception as e:
        print(e)
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/gpt-socket/" + personality,
            eventtype="exception_socket",
            data=str(e),
            session_id=sessionkey,
        )
        if ws.connected:
            ws.send("Error: " + e)
        pass
    # remove the cached session from whrever it is in the cache so it gets reinserted at the tail.
    if sessionkey in __cached_sessions:
        del __cached_sessions[sessionkey]
    # add the session to the tail of the cache
    __cached_sessions[sessionkey] = chat_session
    # shrink the cache if necessary
    while len(__cached_sessions) > session_cache_size:
        del dict[(next(iter(dict)))]
    time.sleep(0.5)
    return ""


@app.route("/describe/<personality>", methods=["GET"])
def describe(personality):
    model_spec = rag.get_model_spec(personality)
    del model_spec["cpu_threads"]
    del model_spec["gpu_layers"]
    del model_spec["flash_attention"]
    del model_spec["intro_dialogue"]
    del model_spec["persona"]
    del model_spec["ui_style"]
    del model_spec["imperative"]
    del model_spec["ui_features"]
    del model_spec["persona_seed"]
    del model_spec["persona_cfg"]
    del model_spec["persona_steps"]
    return model_spec


# Flask route for document embedding
@app.route("/embedding/document", methods=["GET", "POST"])
def embedDocument():
    start = datetime.now()
    returnable = []
    if lock.locked():
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/embedding/document/",
            eventtype="contention",
        )
    with lock:
        llm = getEmbedding(EMBEDDING_MODEL)
        returnable = llm.embed("search_document: " + request.args["text"])
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/embedding/document/",
        eventtype="embedding",
        data=millis_since(start),
    )
    return returnable


@app.route("/embedding/query", methods=["GET", "POST"])
def embedQuery():
    start = datetime.now()
    returnable = []
    if lock.locked():
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/embedding/query",
            eventtype="contention",
        )
    with lock:
        llm = getEmbedding(EMBEDDING_MODEL)
        returnable = llm.embed("search_query: " + request.args["text"])
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/embedding/document/",
        eventtype="embedding",
        data=millis_since(start),
    )
    return returnable


# Flask route for handling tts requests
@app.route("/tts/<personality>", methods=["GET"])
def tts(personality):
    start = datetime.now()
    prompt = request.args["text"]
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/tts/" + personality,
        eventtype="start_speech",
    )
    model_spec = rag.get_model_spec(personality)
    model_path = model_spec["voice"]
    model_param = model_spec["voice_param"]
    filename = str(uuid.uuid1())
    mime_type = "audio/wav"
    if not tts_lock.acquire(blocking=False):
        print("Blocking for TTS lock")
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/tts/" + personality,
            eventtype="contention",
        )
        if not tts_lock.acquire(blocking=True, timeout=120):
            print("Session timeout for TTS")
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/tts/" + personality,
                eventtype="contention_timeout",
            )
            time.sleep(0.5)
            return ""
    # TODO: make this figure out where the Flask process has been invoked from.
    process = subprocess.Popen(
        [
            "../piper/piper",
            "-m",
            model_path,
            "--speaker",
            model_param,
            "-c",
            model_path + ".json",
            "-f",
            filename,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wav, errs = process.communicate(prompt.encode())
    process.wait()
    if COMPRESS_AUDIO_TO_MP3:
        process = subprocess.Popen(
            ["lame", "-m", "s", "--preset", "medium", filename, filename + ".mp3"]
        )
        process.wait()
        os.remove(filename)
        filename = filename + ".mp3"
        mime_type = "audio/mp3"
    f = open(filename, mode="rb")
    data = f.read()
    f.close()
    os.remove(filename)
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/tts/" + personality,
        eventtype="end_speech",
        data=str(datetime.now() - start),
    )
    tts_lock.release()
    return flask.Response(data, mimetype=mime_type)


# Flask route for handling plain old webservice endpoints
@app.route("/gpt/<personality>", methods=["GET", "POST"])
def gpt(personality):
    start = datetime.now()
    prompt = request.args["prompt"]
    messages = []
    session_id = ""
    if "session" in request.args and request.args["session"] in __cached_sessions:
        messages = __cached_sessions.get(request.args["session"], "")
        session_id = request.args["session"]
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/gpt/" + personality,
        eventtype="start_gpt",
        data=prompt,
        session_id=session_id,
    )

    print(prompt)
    responseText = ""
    with lock:
        responseText = flask.Response(
            ask(prompt, personality, chat_context=messages), mimetype="text/plain"
        )
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/gpt/" + personality,
        eventtype="end_gpt",
        data=millis_since(start),
    )
    return responseText


@app.route("/llava/describe", methods=["POST"])
def llava_describe():
    start = datetime.now()
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/llava/describe",
        eventtype="start_description",
    )
    try:
        the_file = io.BytesIO(request.data)
        image = Image.open(the_file, formats=["PNG", "JPEG"])
        output = io.BytesIO()
        image.save(output, "JPEG")
        output.flush()
        output.seek(0)
        base64_utf8_str = base64.b64encode(output.read()).decode("utf-8")
        dataurl = f"data:image/jpeg;base64,{base64_utf8_str}"
        if lock.locked():
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/llava/describe",
                eventtype="contention",
            )

        with lock:
            free_models()
            chat_handler = NanoLlavaChatHandler(
                clip_model_path="../nanollava-mmproj-f16.gguf"
            )
            llm = Llama(
                model_path="../nanollava-text-model-f16.gguf",
                chat_handler=chat_handler,
                n_ctx=4096,
            )
            completion = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that describes images in great detail.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Include as much detail about people as you can. In as much detail as possible, what's in this image?",
                            },
                            {"type": "image_url", "image_url": dataurl},
                        ],
                    },
                ]
            )
            # The abomination on the next line is necessary to force the chat handler to free GPU
            # resources. The built-in catch code only waits for the end of the Python process.
            # chat_handler._llava_cpp.clip_free(chat_handler.clip_ctx)

            del chat_handler
            del llm
            gc.collect()
            returnable = completion["choices"][0]["message"]["content"]
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/llava/describe",
                eventtype="end_description",
                data=millis_since(start),
            )

    except Exception as ex:
        print(ex.with_traceback())
    return returnable


def generate_sd_image_in_fork(model_name, prompt, negativeprompt, steps_value, config_value, seed_value, height, width):
    free_models()
    filename = str(uuid.uuid1()) + ".png"
    process = None
    if "flux" in model_name:
        process = subprocess.Popen(
            [
                "../sd",
                "--diffusion-model",
                "../" + model_name,
                "--vae",
                "../ae-f16.gguf",
                "--clip_l",
                "../clip_l-q8_0.gguf",
                "--t5xxl",
                "../t5xxl_q4_k.gguf",
                "--sampling-method",
                "euler",
                "-p",
                unidecode.unidecode(prompt),
                "-n",
                rag.get_sd_negative_prompt()
                + " "
                + (negativeprompt if negativeprompt else ""),
                "--steps",
                str(steps_value),
                "--cfg-scale",
                str(config_value),
                "-s",
                str(seed_value),
                "-H",
                str(height),
                "-W",
                str(width),
                "-o",
                filename,
            ]
        )
    elif "sd3" in model_name:
        process = subprocess.Popen(
            [
                "../sd",
                "-m",
                "../" + model_name,
                "--sampling-method",
                "euler",
                "-p",
                unidecode.unidecode(prompt),
                "-n",
                rag.get_sd_negative_prompt()
                + " "
                + (negativeprompt if negativeprompt else ""),
                "--steps",
                str(steps_value),
                "--cfg-scale",
                str(config_value),
                "-s",
                str(seed_value),
                "-H",
                str(height),
                "-W",
                str(width),
                "--clip_l",
                "../clip_l.safetensors",
                "--clip_g",
                "../clip_g.safetensors",
                "--t5xxl",
                "../t5xxl_q4_k.gguf",
                "-o",
                filename,
            ]
        )
    else:
        process = subprocess.Popen(
            [
                "../sd",
                "-m",
                "../" + model_name,
                #"--vae",
                #"../sdxl_vae.gguf",
                "-p",
                unidecode.unidecode(prompt),
                "-n",
                rag.get_sd_negative_prompt()
                + " "
                + (negativeprompt if negativeprompt else ""),
                "--steps",
                str(steps_value),
                "--cfg-scale",
                str(config_value),
                "-s",
                str(seed_value),
                "-H",
                str(height),
                "-W",
                str(width),
                "--schedule",
                "karras",
                "-o",
                filename,
            ]
        )
    process.wait()
    f = open(filename, mode="rb")
    output = io.BytesIO(initial_bytes=f.read())
    f.close()
    os.remove(filename)
    return output


def generate_sd_image_in_process(modelName, prompt, steps_value, seed_value, width, height, config_value, negativeprompt, format):
    sd = getSd(modelName)
    images = sd.txt_to_img(
        prompt=unidecode.unidecode(prompt),
        sample_steps=steps_value,
        seed=seed_value,
        width=width,
        height=height,
        cfg_scale=config_value,
        negative_prompt=rag.get_sd_negative_prompt()
        + " "
        + (negativeprompt if negativeprompt else ""),
        sample_method=sd_cpp.stable_diffusion_cpp.SampleMethod.EULER_A,
    )
    image = images[-1]
    output = io.BytesIO()
    if format == "JPEG":
        image = image.convert("RGB")
        image.save(output, "JPEG")
    else:
        image.save(output, "PNG")
    output.flush()
    output.seek(0)
    
    return output


@app.route("/stablediffusion/generate", methods=["GET", "POST"])
def stablediffusion():
    start = datetime.now()
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/stablediffusion/generate",
        eventtype="start_generation",
        data=str(request.args),
    )
    seed_value = random.randint(0, 2**31 - 1)
    steps_value = 20
    config_value = 5
    width = 512
    height = 512
    model_name = "sd.gguf"
    format = "PNG"
    negativeprompt = None
    fork = False

    if "model" in request.args:
        model_name = request.args["model"]
    if "format" in request.args:
        format = request.args["format"]
    if "seed" in request.args:
        seed_value = int(request.args["seed"])
    if "steps" in request.args:
        steps_value = int(request.args["steps"])
    if "width" in request.args:
        width = int(request.args["width"])
    if "cfg" in request.args:
        config_value = float(request.args["cfg"])
    if "height" in request.args:
        height = int(request.args["height"])
    if "negativeprompt" in request.args:
        negativeprompt = request.args["negativeprompt"]
    if "fork" in request.args:
        fork = request.args["fork"].lower() == "true"
    prompt = request.args["prompt"]
    output = None
    if lock.locked():
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/stablediffusion/generate",
            eventtype="contention",
        )

    with lock:
        try:
            output = io.BytesIO()
            if fork:
                output = generate_sd_image_in_fork(model_name, prompt, negativeprompt, steps_value, config_value, seed_value, height, width)
            else:
                output = generate_sd_image_in_process(model_name, prompt, steps_value, seed_value, width, height, config_value, negativeprompt, format)
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/stablediffusion/generate",
                eventtype="end_generation",
                data=millis_since(start),
            )
        except Exception as e:
            print(e)
            log_event(
                username=determine_user(request),
                ip=determine_ip(request),
                subject="/stablediffusion/generate",
                eventtype="exception_generation",
                description=str(e),
            )

    if format == "JPEG":
        return flask.send_file(
            output, mimetype="image/jpeg", download_name="image.jpg", max_age=3600
        )
    return flask.send_file(
        output, mimetype="image/png", download_name="image.png", max_age=3600
    )


def ask(
    prompt, personality="whisper", chat_context=[], force_boolean=False, schema=None
):

    if not lock.locked():
        # The method has been called by a thread not holding the lock.
        raise Exception(
            "Attempted to control the loaded model without holding the exclusivity lock."
        )

    try:
        model_spec = rag.get_model_spec(personality)
        llm = getLlm(personality)
        messages = chat_context.copy()
        if not messages:
            if isSystemlessModel(model_spec["hf_repo"]):
                messages.append(
                    {"role": "user", "content": rag.get_personality_prefix(personality)}
                )
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": rag.get_personality_prefix(personality),
                    }
                )

        messages.append({"role": "user", "content": prompt})

        response_format = None
        if force_boolean:
            response_format = {
                "type": "json_object",
                "schema": {
                    "$id": "https://example.com/person.schema.json",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "is_answer_yes",
                    "type": "boolean",
                },
            }
        elif schema:
            response_format = {"type": "json_object", "schema": schema}

        result = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            stop=stopTokens,
            response_format=response_format,
            max_tokens=model_spec["context_window"],
        )
        print(result["choices"][0]["message"]["content"])
        if schema:
            return json.loads(result["choices"][0]["message"]["content"])
        elif force_boolean:
            return result["choices"][0]["message"]["content"] == "true"
        return result["choices"][0]["message"]["content"]
    except Exception:
        print(traceback.format_exc())
        return ""


def log_event(
    username="anon",
    ip="",
    severity="INFO",
    subject="no_subject",
    eventtype="no_event",
    description="",
    data="",
    session_id="",
):
    with logLock:
        # Open the CSV file in write mode
        with open("log.csv", "a+", newline="") as file:
            writer = csv.writer(file)
            # Write a single row
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    severity,
                    subject,
                    eventtype,
                    description,
                    data,
                    session_id,
                    username,
                    ip,
                ]
            )  # Writing data row


# Get the user identifier for the given request
def determine_user(http_request):
    if "X-Forwarded-User" in http_request.headers:
        return http_request.headers.get("X-Forwarded-User")
    return "anon"


# Get the origin IP for the given request
def determine_ip(http_request):
    if "X-Forwarded-For" in http_request.headers:
        return http_request.headers.get("X-Forwarded-For")
    return http_request.remote_addr


def millis_since(start):
    now = datetime.now()
    span = now - start
    return int(span.total_seconds() * 1000)


# Flask route for handling 'toil' requests having JSON bodies
@app.route("/toil/<personality>", methods=["POST"])
def toil(personality):
    start = datetime.now()
    model_spec = rag.get_model_spec(personality)
    request_context = request.json
    prompt = request_context["prompt"]
    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/toil/" + personality,
        eventtype="start_toil",
        data=prompt,
    )
    rag_text = None
    if "text" in request_context:
        if request_context["text"].startswith("http://") or request_context[
            "text"
        ].startswith("https://"):
            rag_text = rag.fetch_url_text(
                request_context["text"], model_spec["rag_length"]
            )
        else:
            rag_text = request_context["text"]

    if lock.locked():
        log_event(
            username=determine_user(request),
            ip=determine_ip(request),
            subject="/toil/" + personality,
            eventtype="contention",
        )

    with lock:
        llm = getLlm(personality)
        response_format = None
        if "schema" in request_context:
            response_format = {
                "type": "json_object",
                "schema": request_context["schema"],
            }
        # system_prompt = rag.get_personality_prefix(personality)

        if rag_text is not None:
            prompt = (
                "Consider the following text: "
                + rag_text
                + "\n\nGiven the preceding text, "
                + prompt
            )

        messages = []
        if "session" in request.args and request.args["session"] in __cached_sessions:
            messages = __cached_sessions.get(request.args["session"], "").copy()
        else:
            if isSystemlessModel(model_spec["hf_repo"]):
                messages.append(
                    {"role": "user", "content": rag.get_personality_prefix(personality)}
                )
                messages.append({"role": "system", "content": ""})
            else:
                messages.append(
                    {
                        "role": "system",
                        "content": rag.get_personality_prefix(personality),
                    }
                )

        messages.append({"role": "user", "content": prompt})

        result = llm.create_chat_completion(
            messages=messages,
            response_format=response_format,
            temperature=0.7,
            stream=True,
        )

        def generate():
            for tok in result:
                if "content" in tok["choices"][0]["delta"].keys():
                    yield tok["choices"][0]["delta"]["content"]

    log_event(
        username=determine_user(request),
        ip=determine_ip(request),
        subject="/toil/" + personality,
        eventtype="end_toil",
        data=millis_since(start),
    )

    if response_format is None:
        return generate(), {"Content-Type": "text/plain"}
    
    return generate(), {"Content-Type": "application/json"}


app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
# Start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0")
# debug=True
