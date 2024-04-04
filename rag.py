# Import necessary libraries
import feedparser
import requests
import json
import os
from bs4 import BeautifulSoup
from datetime import datetime

# Load personalities from a JSON file
personalities = json.load(open("personalities.json"))

# Initialize a dictionary to cache personalities
personality_cache = {}

# Define default values for various parameters
default_llm_local_file=os.environ.get("LLM_MODEL_FILE", None)
default_llm_hf_repo=os.environ.get("LLM_HUGGINGFACE_REPO", "tsunemoto/bagel-dpo-7b-v0.4-GGUF")
default_llm_hf_filename=os.environ.get("LLM_HUGGINGFACE_FILE", "*Q4_K_M.gguf")
default_llm_gpu_layers=int(os.environ.get("LLM_GPU_LAYERS", "-1")) # -1 for "the whole thing, if supported"
default_llm_context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "2048"))
default_llm_cpu_threads=int(os.environ.get("LLM_CPU_THREADS", "4"))

# Function to get model specification for a given personality
def get_model_spec(personality):
    returnable = {
        'hf_repo': default_llm_hf_repo,
        'hf_filename': default_llm_hf_filename,
        'local_file': default_llm_local_file,
        'gpu_layers': default_llm_gpu_layers,
        'context_window': default_llm_context_window,
        'cpu_threads': default_llm_cpu_threads
    }
    # Update the returnable dictionary with personality-specific values if they exist
    if 'hf_repo' in personalities[personality] :
        returnable['hf_repo'] = personalities[personality]['hf_repo'] 
    if 'hf_filename' in personalities[personality] :
        returnable['hf_filename'] = personalities[personality]['hf_filename']     
    if 'local_file' in personalities[personality] :
        returnable['local_file'] = personalities[personality]['local_file'] 
    if 'gpu_layers' in personalities[personality] :
        returnable['gpu_layers'] = personalities[personality]['gpu_layers'] 
    if 'context_window' in personalities[personality] :
        returnable['context_window'] = personalities[personality]['context_window'] 
    if 'cpu_threads' in personalities[personality] :
        returnable['cpu_threads'] = personalities[personality]['cpu_threads'] 
    return returnable
    
# Function to get the personality prefix
def get_personality_prefix(personality, system_prefix = '', system_suffix = ''):
    # Retrieve the imperative for the given personality
    imperative = personalities[personality]['imperative']
    # Create the personality prefix by concatenating the system prefix, imperative, and system suffix
    personality_prefix = system_prefix + imperative + system_suffix
    return personality_prefix

# Function to get a personality state
def get_personality_state(personality, model, system_prefix = '', system_suffix = ''):
    """Retrieves the model state for the given personality, calculating it if necessary."""
    if(personality not in personality_cache.keys() and personality in personalities.keys() and 'imperative' in personalities[personality].keys()) :
            personality_prefix = get_personality_prefix(personality, system_prefix, system_suffix)
            print("Caching personality for " + personality + "...")
            model.reset()
            model.eval(model.tokenize(personality_prefix.encode()))
            state = model.save_state()
            personality_cache[personality] = state;     
    return personality_cache[personality]

# Function to get RAG prefix
def get_rag_prefix(personality, url, rag_prefix='Consider the following content:\n', rag_suffix='\nGiven the preceding content, ', system_prefix="", system_suffix="", max_url_content_length = 4096, prompt_prefix="", rag_text = None):
    # If a URL is provided, fetch the text from the URL
    if url :
      rag_text = fetchUrlText(url, max_url_content_length)
    # Create the RAG prefix by concatenating the personality prefix, prompt prefix, RAG prefix, RAG text, current date and time, and RAG suffix
    personality_prefix = get_personality_prefix(personality, system_prefix, system_suffix)
    returnable = (personality_prefix + prompt_prefix + rag_prefix + rag_text + getDateTimeText() + rag_suffix) 
    return returnable

# Function to get RAG state
def get_rag_state(personality, model, url, user_prefix = '', rag_prefix='Consider the following content:\n', rag_suffix='\nGiven the preceding content, ', system_prefix="", system_suffix="", max_url_content_length = 4096, rag_text = None):
    """Retrieves a state for the given personality that incorporates the given url as RAG context. The state will be positioned just before the user prompt."""
    model.load_state(get_personality_state(personality, model, system_prefix, system_suffix))
    rag_text = get_rag_prefix(personality, model, url, user_prefix = '', rag_prefix=rag_prefix, rag_suffix=rag_suffix, system_prefix=system_prefix, system_suffix=system_suffix, max_url_content_length = 4096, rag_text=rag_text)
    model.eval(model.tokenize((rag_prefix + rag_text + getDateTimeText() + rag_suffix).encode()))
    state = model.save_state()
    return state

# Function to get the current date and time as a prompt-part
def getDateTimeText():
    now = datetime.now()
    return """\n\n Today's date is {0}. The current time is {1}.""".format(now.strftime("%A, %B %-d, %Y"), now.strftime("%I:%M %p %Z"))

# Function to fetch text from a URL, parsing HTML and feed formats
def fetchUrlText(url, max_url_content_length):
    returnable = ""
    res = requests.head(url)
    # Need to guard against oversized page requests. Not all hosts serve content-length.
    if(res.status_code == 200 and res.headers['content-type'].startswith("text/html") ):
        res = requests.get(url)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        if(soup('main')) :
            returnable = soup.find('main').get_text()
        else :
            returnable = soup.find('body').get_text()
    if(res.status_code == 200 and ( res.headers['content-type'].startswith("application/rss+xml") or res.headers['content-type'].startswith("application/xml") or res.headers['content-type'].startswith("text/xml") or res.headers['content-type'].startswith("application/atom+xml"))):
        feed = feedparser.parse(url)
        returnable += "## "+feed.feed.title + "\n"
        if('description' in feed.feed.keys()):
            returnable += ""+feed.feed.description + "\n"
        for entry in feed.entries :
            returnable += "# " + entry.title + "\n"
            if('description' in entry.keys()) :
                bs = BeautifulSoup(entry.description, features="html.parser")
                returnable += "" + bs.get_text() + "\n"
            if('summary' in entry.keys()) :
                bs = BeautifulSoup(entry.summary, features="html.parser")
                returnable += "" + bs.get_text() + "\n"
            returnable += "\n"
        print(returnable)
    return returnable[:max_url_content_length]
