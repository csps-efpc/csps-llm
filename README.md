# csps-llm
A base for Government of Canada densified Generative AI services like chatbots, RAGs, batch processors, speech synths, and image generators. This is the engine behind the CSPS' "Whisper" menagerie of Generative AI services.

**Please Note:** - the contents of this repo have not (yet) been endorsed by an architecture review board or other governance body. Any use of artificial intelligence by GoC users needs to comply with the appropriate TBS and departmental policies.

### Getting Started

Given a base install of both Python 3 and `pip`, install dependencies with:
```
pip install flask simple_websocket llama_cpp_python stable_diffusion_cpp_python feedparser requests bs4 huggingface_hub duckduckgo_search unidecode plotly pandas wikipedia
```
As an absolute minimum, you'll need 8GB of free RAM and as much local on-disk storage to work with a 7-billion parameter model. To be able to reasonably work with your model, you'll need an AVX2-capable CPU with four real cores (not hyperthreaded).

Once you've got the basics installed, you can start the service with:

```
python app.py
```
You can then browse to `http://localhost:5000/`.

## Model config
The choice of model defaults can be configured at runtime using environment variables:
* `LLM_MODEL_FILE` - path to the GGUF-formatted model to load. If present, overrides any other path settings.
* `LLM_HUGGINGFACE_REPO` - huggingface repo ID from which to load the model
* `LLM_HUGGINGFACE_FILE` - hugginface file reference from which to load the model. Can use wildcards like "*Q4_K_M.gguf".
* `LLM_GPU_LAYERS` - Number of model layers to load onto the GPU. By default, the runtime tries to load all of them if there's a GPU present, or none if there isn't. You only need to set this if you're loading a model that doesn't fit completely on your GPU.
* `LLM_CONTEXT_WINDOW` - size of the context window to allocate - must be equal or less than the maximum context window supported by the model
* `LLM_CPU_THREADS` - Number of hardware threads to allocate to inference. The ideal number is the number of real (ie not SMT, nor hyperthreaded) cores your system has.

*however*

There's a convenient mechanism for declaring several personas in JSON in the folder named `personalities.d`. Over a dozen examples are included in the repo.

### Going fast on NVIDIA

If you have a CUDA-capable GPU, you can make the endpoint use it by following a few additional steps:
* Make sure you have `cmake` installed. Typically `sudo apt install cmake` is all you need.
* Make sure you have NVidia's proprietary drivers installed for your GPU.
* Make sure you have the CUDA toolkit installed. Typically `sudo apt install nvidia-cuda-toolkit`.

Once you've got the prerequisites, reinstall the llama_cpp and stable_diffusion_cpp libraries with CUDA support. This can take quite a while as the upstream project compiles custom CUDA kernels for each kind of layer:

```
CMAKE_ARGS="-DGGML_CUDA=on -DSD_CUBLAS=on" ~/python/bin/pip install llama_cpp_python stable_diffusion_cpp_python --upgrade --force-reinstall --no-cache-dir
```

A device that can do CUDA capability 6.1 or better *really* helps (that's Pascal, GTX 1000, or better). The "Whisper" demo at the CSPS uses a single GTX 1060.

### Going fast on Metal

With kind thanks to our colleagues at Health Canada, the equivalent command on Metal is:
```
CMAKE_ARGS="-DGGML_METAL=ON -DSD_METAL=ON" python/bin/pip install llama_cpp_python stable-diffusion-cpp-python==0.2.1 --upgrade --force-reinstall --no-cache-dir
```
Note that the stable diffusion library is backpinned, because of a defect on Metal in 0.2.2

Brave implementers with other GPUs, extremely new CPUs, or other fancy hardware are encouraged to check out the awesome work at https://github.com/abetlen/llama-cpp-python and to let us know how you make out.

### Personalities
The service supports the creation of an arbitrary number of "personalities" with their own endpoints, which are implemented as system prompts and other tuning parameters. Implementers are encouraged to experiment with their own system prompts, as well as the creation of GPT-like applications by stuffing the personalities with the most common facts that their chatbots are asked for. Most of the models recommended above have an effective window of 6000 tokens for this type of applicaiton, but you can plug in whatever model you like! 

### RAG powers

The websocket endpoint offers certain RAG integrations with public services. Sessions that start with a URL pointing to an HTML page or an RSS/Atom feed will have that URL's content added to the context for the prompt. HTML pages that have accessibility semantics, or feeds with good (short) summaries work best.

# Example prompts:
* "Given the weather forecast, what should I wear tomorrow?"
* "What's the most interesting new learning product from the CSPS?"

Similarly, custom context can be used with if a session begins with something like:
```
|CONTEXT|Your context goes here|/CONTEXT| Your prompt goes here.
```

Sessions starting with `|RAG|somedomainname.com|/RAG|` will search for supporting content among the public-facing content on the given domain. Wikipedia is treated as a special case, using the mediawiki API.

If using the provided UI, the RAG powers can be invoked from the "+" button in the bottom-left corner.

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
    "schema": { An optional https://json-schema.org/ format definition },
    "text": "Either an http URL from which to fetch content, or raw text content to use as context."
}
```
The service will return a well-formed JSON object in every case, and will comply with the provided schema if it's there.
```
/stablediffusion/generate
```
An HTTP GET-only endpoint that accepts prompts and returns images. The two required parameters "prompt" and "seed" set the conditions to be used for generation. An optional "steps" parameter will set how many generation steps will be undertaken, defaulting to 20. The image will be returned in PNG format, unless a "format" parameter with the value "JPEG" is sent. 
Users must place a valid quantized stable diffusion model at the path ../sd.gguf relative to the application directory.

## Contributions
Nothing makes us happier than constructive feedback and well-formed PRs - if you're looking to contribute, some areas where we'd love help:
* Unit and integration tests - as easy or as hard as you like.
* ~~Auto-download of models from HuggingFace - trivial~~
* ~~Add more public RSS/Atom sources as RAG feeds - easy~~
* ~~Add true RAG - medium~~
* ~~Add self-RAG - medium~~
* Make the RSS/Atom RAG retriever configurable at runtime - medium
* Add server-side RAG fact databases - medium -- ~~Initial implementation complete; ie. GPT-style context stuffing.~~
* ~~Add Internet RAG fact databases - medium~~
* ~~Move model configuration to environment variables configured at runtime, with sane defaults - trivial~~
* ~~- AND - Move model configuration to being attributes of "personalities", and make them hot-swap during execution. - easy~~
* Finish the a11y work, particularly around `aria-live`. - moderate
* ~~Support session suspend and restore when several are running concurrently - tricky~~
* ~~Switch to using llama-cpp-python's built in chat formatting - easy~~
* ~~Improved RAG - use the constrained-schema feature to make the model do grounding on the initial prompt and make better decisions about which source(s) to retrieve. - hard trivial, now that the RAG base is complete.~~
* Write a webhook binding for MS Teams - medium
* Write a webhook binding for Slack/Discord - medium
* Make installation auto-detect the best back-end available, and configure it automatically. - hard
* ~~Add a feature to the user interface to expose which model is being used, per TBS guidance. - easy~~
* ~~Set up date formatting to be platform independent - trivial~~
* ~~Bind Stable Diffusion - hard~~
* ~~Bind LLAVA as an image recognizer - hard~~
* ~~Add image sanitizer conditions~~
* ~~Add a clipboard binding for images to the UI. - medium~~
* Add client-media capture of images to the mobile UI. - medium
* ~~Upgrade TTS engine from mimic-3 to Piper~~
* ~~Non-conversational multimodal demo: A Pecha-Kucha generator?~~ - 
* ~~Add unified logging~~
* Add agentic image generation to the conversational UI
* ~~Add trivial user-informed agent example~~
* ~~Move the SPAs to use Flask templates, and the variables into the personalities.~~
* ~~Pechakucha generator.~~
* ~~Document-informed presentation generator.~~
* ~~Add instrumentation for cache hits/misses, and associated timings.~~
* Add personality feature: conversation summarization & reporting
* Personality caching optimization: if calls across personalities use the same model and the same config, they can share cached models all the way to the GPU, although the benefit might be negligible is there's enough RAM to keep the models in MMIO.
* Make choice of personality for pechakucha/explainer/newscast configurable by end-user
* ~~Make personality config files patch one another incrementally~~
* ~~Make diffusion model selectable as a parameter~~
* ~~Add initial stats monitor~~
* Add support for a LRU cache of models (for multi-GPU environments)
* ~~Add support for negative prompts in image generation~~
* Make persona "neutral" avatars auto-generate and cache the first time they're called - cache using a hash?
* ~~Make Wikipedia RAG use the MediaWiki API~~
* Add GCWiki RAG
* ~~Make context-stuffing options capable of calling http, so that contexts can be dynamically stuffed.~~
* ~~Make the /toil endpoint stream responses, even JSON ones.~~
* Add wav2lip generation to speech synth. New endpoint?
* Add "360" use case, where several participants contribute opinions, and the model synthesizes themes.
* ~~Add better error-handling code to the chat UI, addressing network failure cases and the like.~~
* ~~Add better error-handling code to the streaming API, addressing memory issues and context overruns.~~
* ~~Add support for RAG "teaming" - using one model for tool-calling, and another for the subsequent interaction.~~
* Switch utterance detaction and speech recognition to something hybrid client/server-side.
* ~~Make speech synth compress audio to mp3 before sending~~
* ~~Add directory of chat agents~~
* ~~Make auto-detection of LAME on the system happen at startup, and do audio compression only if it's present.~~
* ~~DEFECT: Let upstream uthenticators identify users by name *or* email.~~
* ~~Make image generation endpoint support random seed~~
* ~~Make image generation UI support continuous generation, with several rotating prompts - ensure that seeds advance monotonically, so users can "go back".~~
* Add a RAG feature to expose the authenticated user's name to the prompt.
* ~~DEFECT: If the model directing RAG hallucinates a page entry, recover gracefully.~~
* ~~Introduce guard model and endpoint as demo of alignment~~
* ~~Make Tribble UI filter out orphan subject/object nodes~~
* ~~Add flag to image-generation API endpoint forcing out-of-process generation, and bind to UI.~~
* ~~Strip markdown from text to be spoken in the chat UI.~~
* DEFECT: Visual recognition failed at some point.
* Migrate to kokoro for speech synthesis.
