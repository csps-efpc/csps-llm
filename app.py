import flask
import rag
from flask import redirect, render_template, request
from simple_websocket import Server, ConnectionClosed
from llama_cpp import Llama
import threading

# Initialize the LLM model
app = flask.Flask(__name__) #, static_url_path=''
# Create an exclusive lock on the model state.
lock = threading.Lock()
# Stop tokens
stopTokens = ["### User:","</s>"]
temperature = 0.8
# Prompt parts
system_prefix="### System:\n"
system_suffix="\n"
prompt_prefix = "### User:\n"
prompt_suffix = "\n"
response_prefix = "### Assistant:\n"
response_suffix = "\n"
# Initialize the model
llm = Llama(
        model_path="../neural-chat-7b-v3-3.Q4_K_M.gguf", n_gpu_layers=-1, n_threads=4, numa=True, n_ctx=2048
    )

pleaseWaitText = "\n[Please note that I'm currently helping another user and will be with you as soon as they've finished.]\n"


@app.route("/gpt-socket/<personality>", websocket=True)
def gpt_socket(personality):
    ws = Server.accept(request.environ)
    # We receive and parse the first user prompt.
    message = ws.receive()
    folded = message.casefold()
    url = None
    ## TODO: make this bit modular.
    if(message.startswith("http")):
        s = message.split(" ",1)
        message = s[1]
        url = s[0]
        ws.send("Fetching the page at " + url)
    elif("news".casefold() in folded):
        url = "https://www.cbc.ca/webfeed/rss/rss-topstories"
        if("canad".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-canada"
        if("government".casefold() in folded):
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?sort=publishedDate&orderBy=desc&publishedDate%3E=2021-10-25&pick=100&format=atom&atomtitle=National%20News"
        if("politi".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-politics"
        if("health".casefold() in folded):
            url = "https://api.io.canada.ca/io-server/gc/news/en/v2?topic=health&sort=publishedDate&orderBy=desc&publishedDate%3E=2021-10-25&pick=100&format=atom&atomtitle=Health"
        if("tech".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-technology"
        if("sport".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-sports"
        if("ottawa".casefold() in folded):
            url = "https://www.cbc.ca/webfeed/rss/rss-canada-ottawa"
        ws.send("Checking the news...\n")
    elif(("csps".casefold() in folded or "school".casefold() in folded) and ("cours".casefold() in folded or "learn".casefold() in folded)):
        url = "https://www.csps-efpc.gc.ca/stayconnected/csps-rss-eng.xml"
        ws.send("Checking the catalogue...\n")
    elif(("open".casefold() in folded or "new".casefold() in folded) and "dataset".casefold() in folded):
        url = "https://open.canada.ca/data/en/feeds/dataset.atom"
        ws.send("Checking the open data portal...\n")
    elif(("weather".casefold() in folded or "rain".casefold() in folded  or "snow".casefold() in folded or "temperature".casefold() in folded) and ("today".casefold() in folded or "this week".casefold() in folded or "tomorrow".casefold() in folded or "now".casefold() in folded or "later".casefold() in folded or "forecast".casefold() in folded)):
        url = "https://weather.gc.ca/rss/city/on-118_e.xml"
        ws.send("Checking the forecast...\n")

    try:
        state = None
        if(not lock.acquire(blocking=False)):
            print("Blocking for pre-parsing lock")
            lock.acquire()
        if(url is not None) :
            state = rag.get_rag_state(personality, llm, url, user_prefix=prompt_prefix, system_prefix=system_prefix, system_suffix=system_suffix)
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
            # Get a lock on the model.    
            if(not lock.acquire(blocking=False)):
                print("Blocking for lock")
                lock.acquire()
            llm.load_state(state)    
            llm.eval(llm.tokenize(message.encode()))
            token = llm.sample()
            token_string = llm.detokenize([token]).decode()
            while token is not llm.token_eos():
                print(token_string, end='', flush=True)
                ws.send(token_string)
                accumulator += token_string
                llm.eval([token])
                token = llm.sample()
                token_string = llm.detokenize([token]).decode()
            llm.eval([token]) # Ensure that the model evaluates its own end-of-sequence.
            state = llm.save_state()
            lock.release()
            ws.send("<END>")
            # We wait for a subsequent user prompt, and the cycle begins anew.
            message = prompt_prefix + ws.receive() + prompt_suffix + response_prefix;
    except Exception as e:
        print(e)
        if(lock.locked()):
            lock.release()
        pass
    return ''

# Actually start the flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0")

# debug=True
