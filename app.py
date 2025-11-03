import os
import shutil
import base64
import io
import subprocess
from pathlib import Path

import requests
from PIL import Image
import streamlit as st
import fitz  # PyMuPDF
from diffusers import StableDiffusionPipeline
import torch
import hashlib
from pathlib import PureWindowsPath
from typing import Optional, Tuple
import glob

# --- CONFIG ---
# Choose between using diffusers locally or a local Stable Diffusion HTTP API (e.g. AUTOMATIC1111 or sd-webui)
# Set environment variables to override defaults: SD_MODE (diffusers|api), SD_API_URL, MODEL_NAME
MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/sd-turbo")  # optimized for CPU when available
SD_MODE = os.getenv("SD_MODE", "diffusers")  # or "api"
SD_API_URL = os.getenv("SD_API_URL", "http://127.0.0.1:7860/sdapi/v1/txt2img")

# Ollama settings: the CLI binary and the model name. If you use Ollama, ensure `ollama` is installed and
# the model (e.g. phi3) is available locally or via Ollama's tooling.
OLLAMA_CLI = os.getenv("OLLAMA_CLI", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

CACHE_DIR = Path("image_cache")
CACHE_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_pipeline():
    """Load the diffusers Stable Diffusion pipeline (only used when SD_MODE == 'diffusers')."""
    st.info("Loading Stable Diffusion model... (first time only)")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        pipe = pipe.to("cpu")
        return pipe
    except Exception as e:
        st.error(f"Failed to load diffusers model '{MODEL_NAME}': {e}")
        return None

def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text("text") + "\n"
    return text

def rewrite_prompt_with_ollama(text):
    """Use Ollama local LLM (Phi-3) to turn text into a visual image prompt."""
    base_prompt = f"Rewrite the following passage into a short, vivid, detailed visual art description suitable for an AI image generator:\n\n{text}"
    try:
        # Ensure the Ollama CLI is available
        cli = detect_ollama_cli(OLLAMA_CLI)
        if not cli:
            raise FileNotFoundError(f"'{OLLAMA_CLI}' not found in PATH or common locations")

        # Send the prompt on stdin to avoid shell length issues
        proc = subprocess.run(
            [cli, "run", OLLAMA_MODEL],
            input=base_prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Ollama returned non-zero exit code: {proc.returncode} - {proc.stderr}")

        refined = proc.stdout.strip()
        return refined if refined else text
    except Exception as e:
        st.warning(f"⚠️ Ollama not available or failed: {e}. Using fallback rewriter instead.")
        return fallback_rewrite(text)


def fallback_rewrite(text: str) -> str:
    """A deterministic, local prompt rewriter to use when Ollama isn't available.

    This produces a concise, vivid prompt suitable for SD and is better than raw text.
    """
    # Keep it short and focused: scene + style hints + key objects + mood
    snippet = text.strip().replace('\n', ' ')
    if len(snippet) > 200:
        snippet = snippet[:200].rsplit(' ', 1)[0] + '...'
    template = (
        f"A vivid, cinematic illustration of: {snippet}. "
        "High detail, vibrant colors, dramatic lighting, clear composition, arrows or labels if relevant."
    )
    return template


def detect_ollama_cli(override: Optional[str] = None) -> Optional[str]:
    """Try to locate an `ollama` executable.

    Order of checks:
    - If override is provided and exists, return it
    - shutil.which(override) (or default name)
    - common Windows install paths
    - search under user's ~/.ollama for an ollama.exe
    """
    # 1) explicit override path
    if override:
        if os.path.isfile(override):
            return override

        # maybe user passed a bare name; try which
        which_path = shutil.which(override)
        if which_path:
            return which_path

    # 2) find in PATH using default binary name
    which_default = shutil.which(OLLAMA_CLI)
    if which_default:
        return which_default

    # 3) common locations on Windows
    candidates = [
        str(Path.home() / '.ollama' / 'ollama.exe'),
        str(Path.home() / '.ollama' / 'bin' / 'ollama.exe'),
        str(Path.home() / 'AppData' / 'Local' / 'Programs' / 'Ollama' / 'ollama.exe'),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
    ]
    # Show paths being checked in debug mode
    if st._is_running:
        st.sidebar.write("**Debug: checking paths**")
        for c in candidates:
            exists = os.path.exists(c)
            st.sidebar.text(f"{'✓' if exists else '✗'} {c}")

    for c in candidates:
        if os.path.exists(c):
            return c

    # 4) try a limited rglob search under ~/.ollama (safe small tree)
    try:
        home_ollama = Path.home() / '.ollama'
        if home_ollama.exists() and home_ollama.is_dir():
            matches = list(home_ollama.rglob('ollama.exe'))
            if matches:
                return str(matches[0])
    except Exception:
        pass

    return None


@st.cache_data(ttl=300)
def check_ollama_status(cli_override: Optional[str], model_name: str) -> Tuple[Optional[str], bool, str]:
    """Return (cli_path or None, model_found_bool, message).

    Caches results for a short time to avoid repeated subprocess calls during UI re-renders.
    """
    cli = detect_ollama_cli(cli_override)
    if not cli:
        return None, False, "Ollama CLI not found"

    try:
        proc = subprocess.run([cli, 'list'], capture_output=True, text=True, timeout=8)
        if proc.returncode != 0:
            return cli, False, f"Ollama list failed: {proc.stderr.strip()}"
        out = proc.stdout.lower()
        found = model_name.lower() in out
        msg = "model found" if found else "model not installed"
        return cli, found, msg
    except Exception as e:
        return cli, False, f"Error checking ollama: {e}"

def get_cache_filename(prompt_text):
    """Generate a unique hash filename for each prompt."""
    hash_value = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{hash_value}.png"

def generate_image(prompt_text):
    """Generate or load cached image."""
    cache_path = get_cache_filename(prompt_text)
    if cache_path.exists():
        st.info("✅ Loaded image from cache.")
        return cache_path
    # Two supported generation modes: diffusers (local weights) or an HTTP API (e.g. sd-webui)
    if SD_MODE == "diffusers":
        pipe = load_pipeline()
        if pipe is None:
            st.error("Diffusers pipeline not available. Switch to SD_MODE=api and point SD_API_URL to your local server.")
            raise RuntimeError("Diffusers pipeline unavailable")

        with torch.no_grad():
            image = pipe(prompt_text).images[0]
            image.save(cache_path)
        return cache_path

    elif SD_MODE == "api":
        # Try to call a local Stable Diffusion HTTP API (expects base64 image in JSON response under 'images')
        payload = {
            "prompt": prompt_text,
            "steps": 20,
            "width": 512,
            "height": 512,
        }
        try:
            resp = requests.post(SD_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "images" in data and data["images"]:
                img_b64 = data["images"][0]
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(cache_path)
                return cache_path
            else:
                raise RuntimeError("No images returned from SD API")
        except Exception as e:
            st.error(f"Failed to generate image via SD API at {SD_API_URL}: {e}")
            raise

    else:
        raise ValueError(f"Unsupported SD_MODE: {SD_MODE}")

# --- UI ---
st.set_page_config(page_title="Visual Reader AI", layout="wide")
st.title("📚 Brandon's Reading Buddy for Visual Learners: generate visuals for your favorite pdf.")

# Sidebar configuration (overrides environment variables at runtime)
with st.sidebar.expander("Configuration", expanded=True):
    sd_mode_choice = st.selectbox("Stable Diffusion mode", options=["diffusers", "api"], index=0 if SD_MODE == "diffusers" else 1)
    sd_api_input = st.text_input("SD API URL", value=SD_API_URL)
    model_name_input = st.text_input("Diffusers model name", value=MODEL_NAME)
    ollama_cli_input = st.text_input("Ollama CLI path", value=OLLAMA_CLI)
    ollama_model_input = st.text_input("Ollama model name", value=OLLAMA_MODEL)

# apply sidebar overrides
SD_MODE = sd_mode_choice
SD_API_URL = sd_api_input.strip() or SD_API_URL
MODEL_NAME = model_name_input.strip() or MODEL_NAME
OLLAMA_CLI = ollama_cli_input.strip() or OLLAMA_CLI
OLLAMA_MODEL = ollama_model_input.strip() or OLLAMA_MODEL

# Quick Ollama tests: list installed models and run a small rewrite (useful to verify phi3 is available)
with st.sidebar.expander("Ollama debug / test", expanded=False):
    if st.button("List Ollama models"):
        try:
            if not shutil.which(OLLAMA_CLI):
                st.error(f"'{OLLAMA_CLI}' not found in PATH. Set OLLAMA_CLI to the full path to the ollama executable.")
            else:
                proc = subprocess.run([OLLAMA_CLI, "list"], capture_output=True, text=True, timeout=10)
                if proc.returncode == 0:
                    st.code(proc.stdout)
                else:
                    st.error(f"Ollama list failed: {proc.stderr}")
        except Exception as e:
            st.error(f"Error running ollama list: {e}")

    sample_text = st.text_area("Sample text to rewrite (test)", "A lush forest canopy bathed in sunlight, arrows indicating photosynthesis.")
    if st.button("Run Ollama rewrite"):
        with st.spinner("Running Ollama..."):
            result = rewrite_prompt_with_ollama(sample_text)
            st.write("**Rewrite result:**")
            st.info(result)

# Show Ollama availability status in the main UI
cli_path, model_found, model_msg = check_ollama_status(OLLAMA_CLI, OLLAMA_MODEL)
if cli_path and model_found:
    st.success(f"✓ Ollama CLI found: {cli_path}\n✓ Model '{OLLAMA_MODEL}' available")
elif cli_path and not model_found:
    st.warning(f"✓ Ollama CLI found: {cli_path}\n⚠ Model '{OLLAMA_MODEL}' NOT found ({model_msg})")
else:
    paths_checked = "\n".join(f"- {p}" for p in [
        str(Path.home() / '.ollama' / 'ollama.exe'),
        str(Path.home() / '.ollama' / 'bin' / 'ollama.exe'),
        str(Path.home() / 'AppData' / 'Local' / 'Programs' / 'Ollama' / 'ollama.exe'),
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
    ])
    st.info(f"""
⚠ Ollama CLI not found in common locations:
{paths_checked}

Quick fix:
1. Set this in cmd.exe, then run streamlit:
   set OLLAMA_CLI=C:\\Users\\brand\\AppData\\Local\\Programs\\Ollama\\ollama.exe

2. Or paste the path in the sidebar 'Ollama CLI path' field:
   C:\\Users\\brand\\AppData\\Local\\Programs\\Ollama\\ollama.exe
""")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.session_state["pdf_text"] = text
    st.success("✅ PDF loaded successfully!")

    st.markdown("### 📖 Extracted Text")
    st.text_area("PDF Text (you can copy or select from here):", text, height=300)

    st.markdown("### ✍️ Highlight or paste text to visualize")
    selected_text = st.text_area("Enter or paste text to illustrate:", "", height=100)

    if st.button("🎨 Generate Image"):
        if selected_text.strip():
            with st.spinner("🧠 Rewriting with local LLM (Phi-3)..."):
                visual_prompt = rewrite_prompt_with_ollama(selected_text)
                st.write("**Prompt generated by Phi-3:**")
                st.info(visual_prompt)

            with st.spinner("🎨 Generating or loading image..."):
                image_path = generate_image(visual_prompt)
                st.image(str(image_path), caption="AI-generated illustration", use_column_width=True)
        else:
            st.warning("Please enter text first.")
else:
    st.info("Upload a PDF to begin.")
