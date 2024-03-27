import os
import flask
import rag
from flask import redirect, render_template, request
from simple_websocket import Server, ConnectionClosed
from llama_cpp import Llama
import threading
import traceback
from datetime import datetime

# Initialize the LLM model
app = flask.Flask(__name__) #, static_url_path=''
# Create an exclusive lock on the model state.
lock = threading.Lock()

#Global setting - should we try to cache states?
CACHE_STATES = False
# Stop tokens
stopTokens = ["[/INST]","[INST]","</s>","User:", "Assistant:"]
temperature = 0.8
# Prompt parts
system_prefix="[INST]\n"
system_suffix="\n[/INST]\n"
prompt_prefix = "\nUser: "
prompt_suffix = " \n"
response_prefix = "Assistant: "
response_suffix = ""
rag_prefix = "\nConsider the following:\n"
rag_suffix = "\nGiven the preceding text, "
# Initialize the model

llm_local_file=os.environ.get("LLM_MODEL_FILE", None)
llm_hf_repo=os.environ.get("LLM_HUGGINGFACE_REPO", "tsunemoto/bagel-dpo-7b-v0.4-GGUF")
llm_hf_filename=os.environ.get("LLM_HUGGINGFACE_FILE", "*Q4_K_M.gguf")
llm_gpu_layers=int(os.environ.get("LLM_GPU_LAYERS", "-1")) # -1 for "the whole thing, if supported"

llm = None

if(llm_local_file is not None) :
    llm = Llama(
        model_path=llm_local_file, n_gpu_layers=llm_gpu_layers, n_threads=4, numa=False, n_ctx=2048
    )
else:
    llm = Llama.from_pretrained(
        repo_id=llm_hf_repo,
        filename=llm_hf_filename,
        verbose=False,
        n_gpu_layers=llm_gpu_layers, 
        n_threads=4, 
        numa=False, 
        n_ctx=2048
    )

pleaseWaitText = "\n[Please note that I'm currently helping another user and will be with you as soon as they've finished.]\n"


@app.route("/gpt-socket/<personality>", websocket=True)
def gpt_socket(personality):
    ws = Server.accept(request.environ)
    now = datetime.now()
    time_prompt = """Today's date is {0}. The current time is {1}.
""".format(now.strftime("%A, %B %-d, %Y"), now.strftime("%I:%M %p %Z"))
    # We receive and parse the first user prompt.
    message = ws.receive()
    folded = message.casefold()
    url = None
    text = None
    rag_source_description = ""
    ## TODO: make this bit modular.
    if(message.startswith("|CONTEXT|")):
        s = message[9:].split("""|/CONTEXT|""",1)
        message = s[1]
        text = s[0]
        ws.send("Reading the provided context...")
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
    try:
        if CACHE_STATES:
            state = None
            if(not lock.acquire(blocking=False)):
                print("Blocking for pre-parsing lock")
                lock.acquire()
            if(url is not None) :
                state = rag.get_rag_state(personality, llm, url, rag_text = text, rag_prefix = rag_prefix+rag_source_description, rag_suffix = rag_suffix, user_prefix=prompt_prefix, system_prefix=system_prefix, system_suffix=system_suffix)
            else :
                state = rag.get_personality_state(personality, llm, system_prefix=system_prefix, system_suffix=system_suffix)
                # We tuck the beginning of the user interaction in, because we've got no RAG headers.
                message = prompt_prefix + message
            # At this stage, we're positioned just before the prompt.
            lock.release()
            message += prompt_suffix + response_prefix;

            while True:
                print(message)
                accumulator = '';
                token_bytes = bytearray()
                # Get a lock on the model.    
                if(not lock.acquire(blocking=False)):
                    print("Blocking for lock")
                    lock.acquire()
                llm.load_state(state)    
                llm.eval(llm.tokenize(message.encode()))
                token = llm.sample()
                token_bytes.extend(llm.detokenize([token]))
                while token is not llm.token_eos():
                    try:
                        token_string = token_bytes.decode()
                        print(token_string, end='', flush=True)
                        ws.send(token_string)
                        accumulator += token_string
                        token_bytes = bytearray()
                    except UnicodeError as e:
                        pass # because the token bytes contain an unfinished unicode sequence.
                    llm.eval([token])
                    token = llm.sample()
                    token_bytes.extend(llm.detokenize([token]))
                llm.eval([token]) # Ensure that the model evaluates its own end-of-sequence.
                state = llm.save_state()
                lock.release()
                ws.send("<END>")
                # We wait for a subsequent user prompt, and the cycle begins anew.
                message = prompt_prefix + ws.receive() + prompt_suffix + response_prefix;
        else: #We run the event loop with no cached states, and the lock blocks the llm the whole time.
            chat_session = '';
            if(not lock.acquire(blocking=False)):
                print("Blocking for pre-parsing lock")
                ws.send("(Currently helping another user...)\n")
                lock.acquire()
            if((url is not None) or (text is not None)) :
                chat_session += rag.get_rag_prefix(personality, url, rag_text = text, rag_prefix = rag_source_description, system_prefix=system_prefix, system_suffix=system_suffix)
            else :
                chat_session += rag.get_personality_prefix(personality, system_prefix=system_prefix, system_suffix=system_suffix) + prompt_prefix
            # At this stage, we're positioned just before the prompt.
            chat_session += time_prompt + message + prompt_suffix + response_prefix;
            print(chat_session)
            llm.reset()
            while True:
                stream = llm(
                    chat_session,
                    max_tokens=2048,
                    stop=stopTokens,
                    stream=True,
                    temperature=temperature
                )
                for tok in stream:
                    token_string = tok["choices"][0]["text"]
                    chat_session += token_string
                    print(token_string, end='', flush=True)
                    ws.send(token_string)
                # For some reason - this method doesn't require the EOS token in the stream?!?! chat_session += llm.token_eos()
                ws.send("<END>")
                # We wait for a subsequent user prompt, and the cycle begins anew.
                chat_session += prompt_prefix + ws.receive() + prompt_suffix + response_prefix;
    except Exception as e:
        print(e)
        if(lock.locked()):
            lock.release()
        pass
    return ''

# Plain old webservice endpoints
@app.route("/gpt/<personality>", methods=["GET", "POST"])
def gpt(personality):
    prompt = request.args["prompt"]
    print(prompt)
    lock.acquire()
    llm.reset()
    result=llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": rag.get_personality_prefix(personality),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    lock.release()
    print(result["choices"][0]["message"]["content"])
    return flask.Response(result["choices"][0]["message"]["content"], mimetype="text/plain")


@app.route("/toil/<personality>", methods=["POST"])
def toil(personality):
    request_context = request.json
    prompt = request_context["prompt"]
    rag_text = None
    lock.acquire()
    llm.reset()
    response_format = None
    if('schema' in request_context) :
        response_format={
            "type": "json_object",
            "schema": request_context["schema"]
        }
    system_prompt = rag.get_personality_prefix(personality)
    result=llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
        temperature=0.7,
    )
    lock.release()
    if(response_format is None) :
        return flask.Response(result["choices"][0]["message"]["content"], mimetype="text/plain")
    else:
        return flask.Response(result["choices"][0]["message"]["content"], mimetype="application/json")

# Actually start the flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0")

# debug=True
