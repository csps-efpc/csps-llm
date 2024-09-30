# Import necessary libraries
import feedparser
import requests
import json
import collections
import os
from bs4 import BeautifulSoup
from datetime import datetime

from requests import utils
DEFAULT_USER_AGENT = 'Whisper Agent'
utils.default_user_agent = lambda: DEFAULT_USER_AGENT

# Recursive implementation of dictionary update
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Load personalities from JSON files
personalities = {}
personalities_directory = os.fsencode("./personalities.d")

personality_files = os.listdir(personalities_directory);
personality_files.sort()
print(personality_files)

for personality_file in personality_files:
    personality_filename = os.fsdecode(personality_file)
    if personality_filename.endswith(".json"): 
        personalities = deep_update(personalities, json.load(open("./personalities.d/"+personality_filename)))
        
# Define default values for various parameters
default_llm_local_file=os.environ.get("LLM_MODEL_FILE", None)
default_llm_hf_repo=os.environ.get("LLM_HUGGINGFACE_REPO", "mradermacher/bagel-8b-v1.0-GGUF")
default_llm_hf_filename=os.environ.get("LLM_HUGGINGFACE_FILE", "*Q4_K_M.gguf")
default_llm_gpu_layers=int(os.environ.get("LLM_GPU_LAYERS", "-1")) # -1 for "the whole thing, if supported"
default_llm_context_window=int(os.environ.get("LLM_CONTEXT_WINDOW", "2048"))
default_llm_cpu_threads=int(os.environ.get("LLM_CPU_THREADS", "4"))
default_llm_rag_length=int(os.environ.get("LLM_RAG_LENGTH", "4096"))
default_llm_flash_attention=os.environ.get("LLM_FLASH_ATTENTION", "false")
default_ui_style=os.environ.get("UI_STYLE", "light")
default_ui_features=os.environ.get("UI_FEATURES", "").split(";")
default_llm_voice=os.environ.get("LLM_VOICE", "../en_US-hfc_female-medium.onnx")
default_llm_voice_param=os.environ.get("LLM_VOICE_PARAM", "0")
# A basic set of things we'd prefer not to generate. 
default_sd_negative_prompt=os.environ.get("SD_NEGATIVE_PROMPT", "scary, low quality, extra fingers, mutated hands, watermark, signature")



def get_sd_negative_prompt():
    return default_sd_negative_prompt

def personality_exists(personality):
    return personality in personalities

# Function to get model specification for a given personality
def get_model_spec(personality):
    returnable = {
        'hf_repo': default_llm_hf_repo,
        'hf_filename': default_llm_hf_filename,
        'local_file': default_llm_local_file,
        'gpu_layers': default_llm_gpu_layers,
        'context_window': default_llm_context_window,
        'rag_length': default_llm_rag_length,
        'flash_attention': default_llm_flash_attention,
        'voice': default_llm_voice,
        'voice_param': default_llm_voice_param,
        'ui_style': default_ui_style,
        'ui_features': default_ui_features,
        'cpu_threads': default_llm_cpu_threads,
        'persona': "A purple cat",
        'persona_seed': "2",
        'persona_cfg': "5",
        'persona_steps': "20",
        'intro_dialogue': "",
        'agent_rag_source': None
    }
    # Update the returnable dictionary with personality-specific values if they exist
    returnable.update(personalities[personality])
    return returnable
    
# Function to get the personality prefix
def get_personality_prefix(personality, system_prefix = '', system_suffix = '', include_time = True):
    # Retrieve the imperative for the given personality
    imperative = personalities[personality]['imperative']
    # Create the personality prefix by concatenating the system prefix, imperative, and system suffix
    personality_prefix = system_prefix + imperative
    if(include_time) :
        personality_prefix = personality_prefix + getDateTimeText()
    personality_prefix = personality_prefix + system_suffix
    return personality_prefix

# Function to get RAG prefix
def get_rag_prefix(personality, url, rag_prefix='Consider the following content:\n', rag_suffix='\nGiven the preceding content, ', system_prefix="", system_suffix="", max_url_content_length = 4096, prompt_prefix="", rag_text = None):
    # If a URL is provided, fetch the text from the URL
    if url :
      rag_text = fetchUrlText(url, max_url_content_length)
    # Create the RAG prefix by concatenating the personality prefix, prompt prefix, RAG prefix, RAG text, current date and time, and RAG suffix
    personality_prefix = get_personality_prefix(personality, system_prefix, system_suffix)
    returnable = (personality_prefix + prompt_prefix + rag_prefix + rag_text + rag_suffix) 
    return returnable

# Function to get the current date and time as a prompt-part
def getDateTimeText():
    now = datetime.now()
    return """\n\n Today's date is {0}. The current time is {1}.""".format(now.strftime("%A, %B %d, %Y"), now.strftime("%I:%M %p %Z"))

# Function to fetch text from a URL, parsing HTML and feed formats
def fetchUrlText(url, max_url_content_length):
    returnable = ""
    res = requests.head(url)
    # Need to guard against oversized page requests. Not all hosts serve content-length.
    if(res.status_code == 200 and res.headers['content-type'].startswith("text/html") ):
        res = requests.get(url)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        if(soup.find('div', {'class':"mw-body-content"})) :
            returnable = soup.find('div', {'class':"mw-body-content"}).text
        elif(soup.find('main')) :
            returnable = soup.find('main').get_text()
        else :
            returnable = soup.find('body').get_text()
    if(res.status_code == 200 and ( res.headers['content-type'].startswith("application/rss+xml") or res.headers['content-type'].startswith("application/xml") or res.headers['content-type'].startswith("text/xml") or res.headers['content-type'].startswith("application/atom+xml"))):
        feed = feedparser.parse(url, agent=DEFAULT_USER_AGENT)
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
