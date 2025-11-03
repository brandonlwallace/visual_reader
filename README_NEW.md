# Visual Reader AI (README)

This README_NEW.md contains expanded instructions and config hints for the Visual Reader demo.

See `app.py` for the implementation. The app supports two Stable Diffusion modes:

- `diffusers`: load model with Hugging Face `diffusers` locally.
- `api`: call a local Stable Diffusion HTTP server (e.g. Automatic1111) and expect base64 images in the JSON response under `images`.

Environment variables to configure

- MODEL_NAME (default: stabilityai/sd-turbo)
- SD_MODE (default: diffusers) - set to `api` to use a local server
- SD_API_URL (default: http://127.0.0.1:7860/sdapi/v1/txt2img)
- OLLAMA_CLI (default: ollama)
- OLLAMA_MODEL (default: phi3)

Quick run (Windows cmd.exe):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SD_MODE=api
set SD_API_URL=http://127.0.0.1:7860/sdapi/v1/txt2img
streamlit run app.py
```

Notes
- The app will try to call the Ollama CLI to rewrite selected text; if `ollama` is missing, it will fall back to using the raw text.
- If you plan to use `diffusers` on CPU, expect it to be slow and memory hungry; prefer `api` mode on constrained machines.
