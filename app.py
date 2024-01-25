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
        model_path="../neural-chat-7b-v3-3.Q4_0.gguf", n_gpu_layers=-1, n_threads=4, numa=False, n_ctx=2048
    )

pleaseWaitText = "\n[Please note that I'm currently helping another user and will be with you as soon as they've finished.]\n"


@app.route("/gpt-socket/<personality>", websocket=True)
def gpt_socket(personality):
    ws = Server.accept(request.environ)
    # We receive and parse the first user prompt.
    message = ws.receive()
    
    url = None
    ## TODO: make this bit modular.
    if("news".casefold() in message.casefold()):
        url = "https://www.cbc.ca/webfeed/rss/rss-topstories"
        if("canad".casefold() in message.casefold()):
            url = "https://www.cbc.ca/webfeed/rss/rss-canada"
        if("politi".casefold() in message.casefold()):
            url = "https://www.cbc.ca/webfeed/rss/rss-politics"
        if("tech".casefold() in message.casefold()):
            url = "https://www.cbc.ca/webfeed/rss/rss-technology"
        if("sport".casefold() in message.casefold()):
            url = "https://www.cbc.ca/webfeed/rss/rss-sports"
        ws.send("Checking the news...\n")
    if(("csps".casefold() in message.casefold() or "school".casefold() in message.casefold()) and ("cours".casefold() in message.casefold() or "learn".casefold() in message.casefold())):
        url = "https://www.csps-efpc.gc.ca/stayconnected/csps-rss-eng.xml"
        ws.send("Checking the catalogue...\n")
    if(("open".casefold() in message.casefold() or "new".casefold() in message.casefold()) and "dataset".casefold() in message.casefold()):
        url = "https://open.canada.ca/data/en/feeds/dataset.atom"
        ws.send("Checking the open data portal...\n")
        ws.send("\n")
    try:
        # Get a lock on the model.    
        if(not lock.acquire(blocking=False)):
                ws.send(pleaseWaitText)
                lock.acquire()
        if(url is not None) :
            llm.load_state(rag.get_rag_state(personality, llm, url, user_prefix=prompt_prefix, system_prefix=system_prefix, system_suffix=system_suffix))
        else :
            llm.load_state(rag.get_personality_state(personality, llm, system_prefix=system_prefix, system_suffix=system_suffix))
            # We tuck the beginning of the user interaction in, because we've got no RAG headers.
            message = prompt_prefix + message
        # At this stage, we're positioned just before the prompt.

        message += prompt_suffix + response_prefix;

        while True:
            print(message)
            accumulator = '';
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
            ws.send("<END>")
            # We wait for a subsequent user prompt, and the cycle begins anew.
            message = prompt_prefix + ws.receive() + prompt_suffix + response_prefix;
    except ConnectionClosed:
        pass
    lock.release();

# Actually start the flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0")

# debug=True