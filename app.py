# Import necessary libraries
import gc
import re
import flask
import rag
from flask import redirect, render_template, request
from simple_websocket import Server, ConnectionClosed
from llama_cpp import Llama
from stable_diffusion_cpp import StableDiffusion
import threading
import uuid
import time
import io
from datetime import datetime
from duckduckgo_search import DDGS

# Initialize the Flask app and a thread lock for the LLM model
app = flask.Flask(__name__) #, static_url_path=''
lock = threading.Lock()

# Global settings and constants
stopTokens = ["<|assistant|>", "<|user|>", "<|end|>", "[/INST]","[INST]","</s>","User:", "Assistant:", "[/ASK]", "[INFO]"]
temperature = 0.8
session_cache_size = 100

# Prompt parts
system_prefix="[INST]\n"
system_suffix="\n[/INST]\n"
prompt_prefix = "\nUser: "
prompt_suffix = " \n"
response_prefix = "Assistant: "
response_suffix = ""
rag_prefix = "\nConsider the following:\n"
rag_suffix = "\nGiven the preceding text, "
pleaseWaitText = "\n[Please note that I'm currently helping another user and will be with you as soon as they've finished.]\n"

__cached_llm = None
__cached_personality = None
__cached_sessions = {}

# Function to get the LLM model based on the provided personality
def getLlm(personality):
    global __cached_llm
    global __cached_personality
    if not lock.locked():
        #The method has been called by a thread not holding the lock.
        raise Error('Attempted to control the LLM without holding the exclusivity lock.')
    if(personality == __cached_personality):
    #    __cached_llm.reset()
        return __cached_llm
    if(__cached_llm is not None) :
        del __cached_llm
        gc.collect()
    llm = None
    model_spec = rag.get_model_spec(personality)
    if(model_spec['local_file'] is not None) :
        llm = Llama(
            model_path=model_spec['local_file'], 
            n_gpu_layers=model_spec['gpu_layers'], 
            n_threads=model_spec['cpu_threads'], 
            numa=False, 
            n_ctx=model_spec['context_window'],
            flash_attn=(model_spec['flash_attention'] == 'true')
        )
    else:
        llm = Llama.from_pretrained(
            repo_id=model_spec['hf_repo'],
            filename=model_spec['hf_filename'],
            verbose=True,
            n_gpu_layers=model_spec['gpu_layers'], 
            n_threads=model_spec['cpu_threads'], 
            numa=False, 
            n_ctx=model_spec['context_window'],
            flash_attn=(model_spec['flash_attention'] == 'true')
        )
    __cached_llm = llm
    __cached_personality = personality
    return llm

# Flask route to bounnce users to the default UI
@app.route('/')
def root_redir():
    return redirect("/static/index.html", code=302)
    
# Flask route for handling websocket connections for user conversations
@app.route("/gpt-socket/<personality>", websocket=True)
def gpt_socket(personality):
    ws = Server.accept(request.environ)
    now = datetime.now()
    # We receive and parse the first user prompt.
    message = ws.receive()
    folded = message.casefold()
    url = None
    text = None
    rag_domain = None
    sessionkey = uuid.uuid4().urn
    rag_source_description = ""
    model_spec = rag.get_model_spec(personality)
    ## TODO: make this bit modular.
    if(message.startswith("|SESSION|")):
        s = message[9:].split("""|/SESSION|""",1)
        message = s[1]
        sessionkey = s[0]
    elif(message.startswith("|CONTEXT|")):
        s = message[9:].split("""|/CONTEXT|""",1)
        message = s[1]
        text = s[0]
        ws.send("Reading the provided context...")
    elif(message.startswith("|RAG|")):
        s = message[5:].split("""|/RAG|""",1)
        message = s[1]
        rag_domain = s[0]
        reflection = ask("If the following text is a question that could be answered with a web search, answer with a relevant search term. Answer with only the terms surrounded by \" characters, or \"none\" if the question is inappropriate. Do not answer anything after the quoted string.\n\n" + message, personality)
        print(reflection)
        matches = re.search(r'"([^"]+)"', reflection)
        if(matches is not None and matches.group(1) != "none") :
            ws.send("Searching for \""+matches.group(1)+"\" ... ")
            query = matches.group(1) + " site:" + (" OR site:".join(rag_domain.split('|')))
            results = DDGS().text(query)
            if(results) :
                for i, result in enumerate(results, start = 1) :
                    if(i < 3 and text is None) :
                        possible_text = rag.fetchUrlText(results[i]['href'], model_spec['rag_length'])
                        ws.send("Evaluating search result "+ str(i) + "...")
                        ws.send("\n\n")
                        self_eval = ask("Consider the following content: \n\n" + possible_text + "\n\nDoes the content answer the request \""+message+"\"?", force_boolean=True)
                        if(self_eval == 'true') :
                            ws.send("Content is relevant: ["+ results[i]['title'] +"](" + results[i]['href'] + "):")
                            ws.send("\n\n")
                            rag_source_description = "The page \""+results[i]['title']+"\" at "+ results[i]['href'] +" says:\n"
                            text = possible_text
                            print("Accepting the page \""+results[i]['title']+"\" at "+ results[i]['href'] +" as relevant.")
                        else:
                            print("Discarding the page \""+results[i]['title']+"\" at "+ results[i]['href'] +" as irrelevant.")
            else:
                ws.send("Couldn't find a suitable reference.")
                ws.send("\n\n")
        else:
            ws.send("No search necessary.")
            ws.send("\n\n")
    elif(message.startswith("http")):
        s = message.split(" ",1)
        message = s[1]
        url = s[0]
        ws.send("Fetching the page at " + url)
    elif("news".casefold() in folded):
        rag_source_description = "The latest news from the CBC is:\n"
        url = "https://www.cbc.ca/webfeed/rss/rss-topstories"
        if("canad".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-canada"
        if("government".casefold() in folded):
            rag_source_description = "The latest news from the Government of Canada is:\n"
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?sort=publishedDate&orderBy=desc&pick=5&format=atom&atomtitle=National%20News"
        if("politi".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-politics"
        if("health".casefold() in folded):
            rag_source_description = "The latest health news from the Government of Canada is:\n"
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?topic=health&sort=publishedDate&orderBy=desc&pick=5&format=atom&atomtitle=Health"
        if("tech".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-technology"
        if("gaming".casefold() in folded):
            url = "https://kotaku.com/rss"
        if("sport".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-sports"
        if("financ".casefold() in folded):
            url = "https://financialpost.com/feed"
        if("ottawa".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-canada-ottawa"
        if("ottawa citizen".casefold() in folded):
            url = "https://ottawacitizen.com/feed"
        ws.send("Checking the news...\n")
    elif("busrides".casefold() in folded):
        rag_source_description = "Busrides is a series of articles from the CSPS. Busrides is not about riding public transit. What follows is a list of the newest article titles and summaries from Busrides:\n"
        url = "https://busrides.ghost.io/rss/"
        ws.send("Checking the latest microlearning...\n")

    elif(("csps".casefold() in folded or "school".casefold() in folded) and ("cours".casefold() in folded or "learn".casefold() in folded)):
        rag_source_description = "The latest learning products from the CSPS are:\n"
        url = "https://www.csps-efpc.gc.ca/stayconnected/csps-rss-eng.xml"
        ws.send("Checking the catalogue...\n")
    elif(("open".casefold() in folded or "new".casefold() in folded) and "dataset".casefold() in folded):
        rag_source_description = "The latest open datasets from the Government of Canada are:\n"
        url = "https://open.canada.ca/data/en/feeds/dataset.atom"
        ws.send("Checking the open data portal...\n")
    elif(("weather".casefold() in folded or "rain".casefold() in folded  or "snow".casefold() in folded or "temperature".casefold() in folded) and ("today".casefold() in folded or "this week".casefold() in folded or "tomorrow".casefold() in folded or "now".casefold() in folded or "later".casefold() in folded or "forecast".casefold() in folded)):
        rag_source_description = "The weather forecast for the Ottawa area is:\n"
        url = "https://weather.gc.ca/rss/city/on-118_e.xml"
        ws.send("Checking the forecast...\n")
    elif(("nouvelles").casefold() in folded):
        rag_source_description = "Les dernieres nouvelles de La Presse sont :\n"
        url = "https://www.lapresse.ca/actualites/rss"
        ws.send("Je rassemble les actualités...\n")
    

    chat_session = []

    try:
        if(sessionkey in __cached_sessions) :
            chat_session = __cached_sessions.get(sessionkey, '')
            chat_session.append({"role": "user", "content": message})
        else :
            first_prompt = ''
            if('Phi-3' in model_spec['hf_repo'] or 'Mistral-7B-Instruct-v0.3' in model_spec['hf_repo']) : 
                chat_session.append({"role": "user", "content": rag.get_personality_prefix(personality)})
            else:
                chat_session.append({"role": "system", "content": rag.get_personality_prefix(personality)})
            if(url is not None):
                text = rag.fetchUrlText(url, model_spec['rag_length'])
            if(text is not None) :
                first_prompt += 'Consider the following content:\n' + text + '\nGiven the preceding content, '
            first_prompt += message
            chat_session.append({"role": "user", "content": first_prompt})

        if(not lock.acquire(blocking=False)):
            print("Blocking for pre-parsing lock")
            ws.send("(Currently helping another user...)")
            ws.send("\n\n")
            if( not lock.acquire(blocking=True, timeout=120)) :
                ws.send("Error: Couldn't get the model's attention for 120s.")
                ws.send("<END "+sessionkey+">")
                ws.send(" ") #Junk frame to ensure the previous one gets flushed?
                print("Session timeout " + sessionkey)
                time.sleep(0.5)
                return ''
        llm = getLlm(personality)
        
        print(chat_session)
        stream = llm.create_chat_completion(
            chat_session,
            max_tokens=model_spec['context_window'],
            stop=stopTokens,
            stream=True,
            temperature=temperature
        )
        response =''
        for tok in stream:
            if(("content" in tok["choices"][0]['delta'].keys())) :
                token_string = tok["choices"][0]['delta']["content"]
                response += token_string
                print(token_string, end='', flush=True)
                ws.send(token_string)
        # For some reason - this method doesn't require the EOS token in the stream?!?! chat_session += llm.token_eos()
        ws.send("<END "+sessionkey+">")
        ws.send(" ") #Junk frame to ensure the previous one gets flushed?
        print("End session " + sessionkey)
        # We prepare the session for a subsequent user prompt, and the cycle begins anew.
        chat_session.append({"role": "assistant", "content": response});
        # TODO: if and when the cache moves out of process memory to a K-V store, the chat session will need to get written to that cache.
    except Exception as e:
        print(e)
        pass;
    if(lock.locked()):
        lock.release()
    # remove the cached session from whrever it is in the cache so it gets reinserted at the tail.
    if(sessionkey in __cached_sessions):
        del __cached_sessions[sessionkey]
    # add the session to the tail of the cache
    __cached_sessions[sessionkey] = chat_session
    # shrink the cache if necessary
    while(len(__cached_sessions) > session_cache_size):
        del dict[(next(iter(dict)))]
    time.sleep(0.5)
    return ''

# Flask route for handling plain old webservice endpoints
@app.route("/gpt/<personality>", methods=["GET", "POST"])
def gpt(personality):
    prompt = request.args["prompt"]
    print(prompt)
    return flask.Response(ask(prompt, personality), mimetype="text/plain")

@app.route("/stablediffusion/generate", methods=["GET", "POST"])
def stablediffusion():
    global __cached_llm
    seed = request.args["seed"]
    seed_value = 42
    prompt = request.args["prompt"]
    output = None
    if(seed) :
        seed_value = int(seed)
    if(not lock.acquire(blocking=False)):
            print("Blocking for pre-parsing lock")
            if( not lock.acquire(blocking=True, timeout=120)) :
                print("Session timeout for image generation")
                time.sleep(0.5)
                return ''
    # ensure that the LLM model is unloaded for the duration of the lock.
    if(__cached_llm is not None) :
        del __cached_llm
        gc.collect()
    sd = None
    try:
        #Instantiate the model
        sd = StableDiffusion(
            model_path="../sd.gguf",
            vae_path="../sdxl_vae.safetensors"
        )
        images = sd.txt_to_img(
            prompt=prompt,
            sample_steps = 20,
            seed=seed_value
        )
        output = io.BytesIO()
        images[-1].save(output, "PNG")
        output.flush()
        output.seek(0)
    except Exception as e:
        print(e)
        pass;
    if(sd):
        #Clear the stable diffusion model from memory
        del sd
        gc.collect()

    if(lock.locked()):
        lock.release()

    return flask.send_file(output, mimetype="image/png", download_name="image.png")

def ask(prompt, personality="whisper", chat_context = [], force_boolean = False):
    lock.acquire()
    model_spec = rag.get_model_spec(personality)
    llm=getLlm(personality)
    messages = chat_context.copy()
    if(not messages) :
        if('Phi-3' in model_spec['hf_repo'] or 'Mistral-7B-Instruct-v0.3' in model_spec['hf_repo']) : 
            messages.append({"role": "user", "content": rag.get_personality_prefix(personality)})
        else:
            messages.append({"role": "system", "content": rag.get_personality_prefix(personality)})

    messages.append({"role": "user", "content": prompt})

    response_format = None
    if(force_boolean) :
        response_format = {
            "type": "json_object",
            "schema": {
                "$id": "https://example.com/person.schema.json",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "is_answer_yes",
                "type": "boolean"
            }
        }

    result=llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        stop=stopTokens,
        response_format=response_format        
    )
    lock.release()
    return result["choices"][0]["message"]["content"]

# Flask route for handling 'toil' requests having JSON bodies
@app.route("/toil/<personality>", methods=["POST"])
def toil(personality):
    request_context = request.json
    prompt = request_context["prompt"]
    rag_text = None
    if('text' in request_context):
        rag_text = request_context['text']
    lock.acquire()
    model_spec = rag.get_model_spec(personality)
    llm=getLlm(personality)
    response_format = None
    if('schema' in request_context) :
        response_format={
            "type": "json_object",
            "schema": request_context["schema"]
        }
    system_prompt = rag.get_personality_prefix(personality)

    if(rag_text is not None):
        prompt = "Consider the following text: " + rag_text + "\n\nGiven the preceding text, " + prompt

    messages = []

    if('Phi-3' in model_spec['hf_repo'] or 'Mistral-7B-Instruct-v0.3' in model_spec['hf_repo']) : 
        messages.append({"role": "user", "content": rag.get_personality_prefix(personality)})
    else:
        messages.append({"role": "system", "content": rag.get_personality_prefix(personality)})

    messages.append({"role": "user", "content": prompt})

    result=llm.create_chat_completion(
        messages=messages,
        response_format=response_format,
        temperature=0.7,
    )
    lock.release()
    if(response_format is None) :
        return flask.Response(result["choices"][0]["message"]["content"], mimetype="text/plain")
    else:
        return flask.Response(result["choices"][0]["message"]["content"], mimetype="application/json")

# Start the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0")

# debug=True

