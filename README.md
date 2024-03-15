# csps-llm
A base for Government of Canada LLM services like chatbots, RAGs, and batch processors. This is the engine behind the CSPS' "Whisper" and "Chuchotte" demo services.

### Getting Started

Given a base install of both Python 3 and `pip`, install dependencies with:
```
pip install flask simple_websocket traceback llama_cpp feedparser requests json bs4
```

You'll need to download a model to use. We recommend one of the Q4 versions of https://huggingface.co/tsunemoto/bagel-dpo-7b-v0.4-GGUF/tree/main although Mistral-instruct, Mixtral-instruct, and Intel Neural Chat all work extremely well. By default, the app loads the model from the parent of the working directory.

As an absolute minimum, you'll need 12GB of RAM and as much local on-disk storage to work with a 7-billion parmeter model. To be able to reasonably work with your model, you'll need an AVX2-capable CPU with four real cores (not hyperthreaded).

Once you've got the basics installed, you can start the service with:

```
python app.py
```
and then browse to `http://localhost:5000/static/index.html`.

### Going fast

If you have a CUDA-capable GPU, you can make the endpoint use it by following a few additional steps:
* Make sure you have `cmake` installed. Typically `sudo apt install cmake` is all you need.
* Make sure you have NVidia's proprietary drivers installed for your GPU.
* Make sure you have the CUDA toolkit installed. Typically `sudo apt install nvidia-cuda-toolkit`.

Once you've got the prerequisites, reinstall the llama_cpp library with CUDA support:

```
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

A device that can do CUDA capability 6.1 or better *really* helps (that's Pascal, GTX 1000, or better). The "Whisper" demo at the CSPS uses a single GTX 1070.

Brave implementers with other GPUs, extremely new CPUs, or other fancy hardware are encouraged to check out the awesome work at https://github.com/abetlen/llama-cpp-python and to let us know how you make out.

### Personalities
The service supports the creation of an arbitrary number of "personalities" with their own endpoints, which are implemented as system prompts. Implementers are encouraged to experiment with their own system prompts, as well as the creation of GPT-like applications by stuffing the personalities with the most common facts that their chatbots are asked for. The models recommended above have an effective window of 4000 tokens for this type of applicaiton, but you can plug in whatever model you like! 

### RAG powers

The websocket endpoint offers certain RAG integrations with public services. Sessions that start with a URL pointing to an HTML page or an RSS/Atom feed will have that URL's content added to the context for the prompt. HTML pages that have accessibility semantics, or feeds with good (short) summaries work best.

# Example prompts:
* "Given the weather forecast, what should I wear tomorrow?"
* "What's the most interesting new learning product from the CSPS?"
* "What are the latest articles about AI on Busrides?"

Similarly, custom context can be used with if a session begins with something like:
```
|CONTEXT|Your context goes here|/CONTEXT| Your prompt goes here.
```
## Endpoints
The service offers three endpoints:
```
/gpt-socket/<personality>
```
A websocket endpoint with a dead-simple contract: Frames sent to the server are treated as prompts, frames returned from the server are tokens that can be concatenated into a response. The end of generation is marked by a frame containing `<END>`.

```
/gpt/<personality>
```
An HTTP endpoint that accepts URL-encoded requests over GET and POST with a single parameter: "prompt". Returns the generated response as a string.

```
/toil/<personality>
```
An HTTP POST-only endpoint that accepts JSON-coded requests of the form:
```json
{
    "prompt": "The prompt for the model.",
    "schema": { An optional https://json-schema.org/ format definition }
}
```
The service will return a well-formed JSON object in every case, and will comply with the provided schema if it's there.

## Contributions
Nothing makes us happier than constructive feedback and well-formed PRs - if you're looking to contribute, some areas where we'd love help:
* Unit and integration tests - as easy or as hard as you like.
* Auto-download of models from HuggingFace - trivial
* Add more public RSS/Atom sources as RAG feeds - easy
* Add server-side RAG fact databases - medium
* Move model configuration to environment variables configured at runtime, with sane defaults - trivial
* - OR - Move model configuration to being attributes of "personalities", and make them hot-swap during execution. - easy
* Finish the a11y work, particularly around `aria-live`. - moderate
* Support session suspend and restore when several are running concurrently - tricky
* Switch to using llama-cpp-python's built in chat formatting - easy
* Improved RAC - use the constrained-schema feature to make the model do grounding on the initial prompt and make better decisions about which source(s) to retrieve. - hard
* Write a webhook binding for MS Teams - medium
* Write a webhook binding for Slack/Discord - medium
* Make installation auto-detect the best back-end available, and configure it automatically. - hard
