import feedparser
import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime

personalities = json.load(open("personalities.json"))

personality_cache = {}

def get_personality_state(personality, model, system_prefix = '', system_suffix = ''):
    """Retrieves the model state for the given personality, calculating it if necessary."""
    if(personality not in personality_cache.keys() and personality in personalities.keys()) :
        imperative = ""
        if(personality in personalities.keys() and 'imperative' in personalities[personality].keys()) :
            print("Calculating personality for " + personality + "...")
            imperative = personalities[personality]['imperative']
            model.reset()
            print (system_prefix + imperative + system_suffix)
            model.eval(model.tokenize((system_prefix + imperative + system_suffix).encode()))
            state = model.save_state()
            personality_cache[personality] = state;     
    return personality_cache[personality]

def get_rag_state(personality, model, url, user_prefix = '', rag_prefix='Consider the following content:\n', rag_suffix='\nGiven the preceding content, ', system_prefix="", system_suffix="", max_url_content_length = 4096):
    """Retrieves a state for the given personality that incorporates the given url as RAG context. The state will be positioned just before the user prompt."""
    model.load_state(get_personality_state(personality, model, system_prefix, system_suffix))
    rag_text = fetchUrlText(url, max_url_content_length)
    print ((rag_prefix + rag_text + getDateTimeText() + rag_suffix))
    model.eval(model.tokenize((rag_prefix + rag_text + getDateTimeText() + rag_suffix).encode()))
    state = model.save_state()

def getDateTimeText():
    now = datetime.now()
    return """\n\n Today's date is {0}. The current time is {1}.""".format(now.strftime("%A, %B %-d, %Y"), now.strftime("%I:%M %p %Z"))

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
