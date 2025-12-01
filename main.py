import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import platform
import sys
from typing import Tuple, Union

# Helper for runtime type checks
def _ensure_type(name: str, value, expected: Union[type, Tuple[type, ...]]):
    if not isinstance(value, expected):
        expected_names = (
            expected.__name__ if isinstance(expected, type) else 
            ", ".join(t.__name__ for t in expected)
        )
        raise TypeError(f"Invalid type for '{name}': expected {expected_names}, got {type(value).__name__}")
import os
from datetime import datetime
import traceback


def _log_error(mode: str, message: str, exc: Exception = None):
    """Append an error entry to `data/logs/errors.log` with timestamp and mode.

    This helper is intentionally tolerant: failures to write the log are
    ignored to avoid cascading errors in the main application flow.
    """
    try:
        log_dir = os.path.join(os.getcwd(), "data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "errors.log")
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{mode}] {message}\n")
            if exc is not None:
                f.write(traceback.format_exc())
                f.write("\n")
    except Exception:
        # Silently ignore logging failures to avoid cascading errors
        pass

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device detected: {DEVICE}")
print(f"System: {platform.system()} {platform.machine()}")
print(f"Python: {sys.version.split()[0]}")


# Alternative: use quantized versions if loading issues
# TEXT_MODEL = "ggml-org/SmolLM3-3B-GGUF"  # Quantized version
# VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Global variables for models
text_tokenizer = None
vision_model = None
vision_processor = None

DEVICE = 'cpu'
VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def load_vision_model():
    """Load SmolVLM2 vision model"""
    global vision_model, vision_processor

    if vision_model is None:
        print(f"Loading {VISION_MODEL}...")
        try:
            # Load processor with trust_remote_code
            vision_processor = AutoProcessor.from_pretrained(
                VISION_MODEL,
                trust_remote_code=True
            )

            # Optimized loading based on device
            if DEVICE == "cuda":
                try:
                    vision_model = AutoModelForImageTextToText.from_pretrained(
                        VISION_MODEL,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                except:
                    # Fallback without device_map if accelerate is not available
                    vision_model = AutoModelForImageTextToText.from_pretrained(
                        VISION_MODEL,
                        dtype=torch.float16,
                        trust_remote_code=True
                    ).to(DEVICE)
            elif DEVICE == "mps":
                vision_model = AutoModelForImageTextToText.from_pretrained(
                    VISION_MODEL,
                    dtype=torch.float16,
                    trust_remote_code=True
                ).to(DEVICE)
            else:
                vision_model = AutoModelForImageTextToText.from_pretrained(
                    VISION_MODEL,
                    dtype=torch.float32,
                    trust_remote_code=True
                ).to(DEVICE)

            print("SmolVLM2 loaded successfully!")
        except Exception as e:
            print(f"Error loading SmolVLM2: {e}")
            print("\nMake sure you have the correct version:")
            print("   pip install --upgrade transformers>=4.45.0")
            raise

    return vision_model, vision_processor

def apply_settings(selected_device: str, selected_model: str):
    """Apply device and model selection from the UI.

    - selected_device: one of 'auto', 'cuda', 'cpu', 'mps'
    - selected_model: HF model id for vision
    """
    global DEVICE, VISION_MODEL

    # Determine device
    if selected_device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        # Accept only known strings
        DEVICE = selected_device

    if selected_model:
        VISION_MODEL = selected_model

    info = (
        f"Applied settings:\nDevice: {DEVICE}\n Model: {VISION_MODEL}"
    )

    return info

def analyze_image(image, question, max_length=256):
    """Analyze an image with SmolVLM2"""
    try:
        if image is None:
            return "Please upload an image."

        try:
            _ensure_type("question", question, str)
            _ensure_type("max_length", max_length, int)
        except TypeError as e:
            return f"TypeError: {e}"

        if not question or question.strip() == "":
            return "Please ask a question about the image."

        # Basic validation for image type (accept PIL.Image or numpy array-like)
        if not isinstance(image, Image.Image):
            try:
                import numpy as _np
                if not isinstance(image, _np.ndarray):
                    return "Invalid image type: expected a PIL Image or numpy array."
            except Exception:
                return "Invalid image type: expected a PIL Image or numpy array."

        model, processor = load_vision_model()

        # Prepare image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Format prompt for SmolVLM2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Process image and text
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False
            )

        # Decoding
        generated_text = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        elif question in generated_text:
            response = generated_text.split(question)[-1].strip()
        else:
            response = generated_text

        return response if response else generated_text

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Full error:\n{error_detail}")

        # Detailed error message for user
        error_msg = f"Error: {str(e)}\n\n"

        if "Unrecognized processing class" in str(e):
            error_msg += "Solution: Update transformers:\n"
            error_msg += "   pip install --upgrade transformers>=4.45.0\n\n"

        error_msg += "Check the console for more details."
        return error_msg

# Interface Gradio
with gr.Blocks(title="SmolVLM2", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🤖 SmolVLM2 - Multimodal Interface

    Complete interface for text analysis and image processing with **SmolVLM2** models from HuggingFace.

    **Features:**
    - 👁️ Image Description
    - 📝 OCR from images
    - Compatible with CPU, CUDA GPU, and Apple Silicon (MPS)
    """)
    
    with gr.Tabs():
        
        # Vision Mode Tab
        with gr.Tab("👁️ Vision Mode"):
            gr.Markdown("### Image analysis with SmolVLM2")
            
            with gr.Row():
                with gr.Column():
                    vision_image = gr.Image(
                        type="pil",
                        label="Upload an image"
                    )
                    vision_question = gr.Textbox(
                        label="Your question about the image",
                        placeholder="Describe this image in detail...",
                        value="Describe this image.", 
                        lines=3
                    )
                    vision_max_length = gr.Slider(
                        minimum=50,
                        maximum=512,
                        value=256,
                        step=50,
                        label="Max response length"
                    )
                    vision_submit = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
                    vision_output = gr.Textbox(
                        label="Analysis",
                        lines=15,
                        interactive=False
                    )
                    vision_error_modal = gr.HTML(visible=False)
                    vision_save_btn = gr.Button("💾 Save Output")
                    vision_download = gr.File(label="Download Analysis", visible=False)

            # Clear button after component definition
            with gr.Row():
                clear_vision_btn = gr.Button("🗑️ Clear All")
                clear_vision_btn.click(
                    fn=lambda: (None, "", ""),
                    inputs=[],
                    outputs=[vision_image, vision_question, vision_output]
                )
        
            
            def analyze_image_safe(image, question, max_length=256):
                """Call analyze_image and return (text, modal_update).

                If analyze_image returns an error-like string (starts with ❌, ⚠️, or contains 'TypeError'),
                show a modal with guidance.
                """
                try:
                    resp = analyze_image(image, question, max_length)
                except Exception as e:
                    modal_html = (
                        f"<div style='padding:16px;background:#fff;border-radius:8px;'>"
                        f"<h3 style='color:#c00;margin:0 0 8px;'>Ошибка</h3>"
                        f"<p>Произошла внутренняя ошибка: {e}</p>"
                        "</div>"
                    )
                    return f"❌ Внутренняя ошибка: {e}", gr.HTML.update(value=modal_html, visible=True)

                # If analyze_image returned an error message, show modal
                if isinstance(resp, str) and (resp.startswith("❌") or resp.startswith("⚠️") or "TypeError" in resp):
                    modal_html = (
                        "<div style='padding:16px;background:#fff;border-radius:8px;'>"
                        f"<h3 style='color:#c00;margin:0 0 8px;'>Ошибка ввода</h3>"
                        f"<p>{resp}</p>"
                        "<p>Проверьте формат и повторите попытку.</p>"
                        "</div>"
                    )
                    return resp, gr.HTML.update(value=modal_html, visible=True)

                return resp, gr.HTML.update(value="", visible=False)

            vision_submit.click(
                fn=analyze_image_safe,
                inputs=[vision_image, vision_question, vision_max_length],
                outputs=[vision_output, vision_error_modal]
            )

            def _save_text_to_file(text, prefix="output"):
                if not text or (isinstance(text, str) and text.strip() == ""):
                    return None

                out_dir = os.path.join(os.getcwd(), "data", "outputs")
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                filename = f"{prefix}_{ts}.txt"
                path = os.path.join(out_dir, filename)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                    return path
                except Exception as e:
                    print(f"Error saving file: {e}")
                    return None

            def save_vision_output(text):
                return _save_text_to_file(text, prefix="vision_analysis")

            vision_save_btn.click(
                fn=save_vision_output,
                inputs=[vision_output],
                outputs=[vision_download]
            )

        # OCR Tab (using SmolVLM)
        with gr.Tab("📄 OCR"):
            gr.Markdown("""### Optical Character Recognition (OCR) with SmolVLM
            """)

            with gr.Row():
                with gr.Column():
                    ocr_image = gr.Image(type="pil", label="Upload an image")
                    ocr_lang = gr.Dropdown(
                        choices=["auto", "eng", "rus", "eng+rus"],
                        value="auto",
                        label="Language hint (optional)"
                    )
                    ocr_max_length = gr.Slider(
                        minimum=50,
                        maximum=1024,
                        value=256,
                        step=50,
                        label="Max response length"
                    )
                    ocr_submit = gr.Button("📄 Run OCR", variant="primary")

                with gr.Column():
                    ocr_output = gr.Textbox(label="Detected text", lines=15, interactive=False)
                    ocr_error_modal = gr.HTML(visible=False)
                    ocr_save_btn = gr.Button("💾 Save Output")
                    ocr_download = gr.File(label="Download OCR", visible=False)

            def ocr_via_analyze(image, lang_hint, max_length=256):
                """Wrapper that reuses the existing `analyze_image` function for OCR.

                It builds a focused OCR question and calls `analyze_image`, so model
                loading and processing remain centralized.
                """
                if image is None:
                    return "⚠️ Please upload an image."

                # Validate types for OCR wrapper
                try:
                    _ensure_type("lang_hint", lang_hint, (str, type(None)))
                    _ensure_type("max_length", max_length, int)
                except TypeError as e:
                    return f"TypeError: {e}"

                lang_note = "" if (not lang_hint or lang_hint == "auto") else f" Language hint: {lang_hint}."
                question = (
                    "Extract all visible text from the provided image.\n"
                    "Return only the raw text found in the image, preserving line breaks and number formatting."
                    + lang_note
                )

                # Delegate to existing analyze_image (which uses SmolVLM)
                return analyze_image(image, question, max_length)

            def ocr_via_analyze_safe(image, lang_hint, max_length=256):
                try:
                    resp = ocr_via_analyze(image, lang_hint, max_length)
                except Exception as e:
                    modal_html = (
                        "<div style='padding:16px;background:#fff;border-radius:8px;'>"
                        f"<h3 style='color:#c00;margin:0 0 8px;'>Ошибка</h3>"
                        f"<p>Произошла внутренняя ошибка: {e}</p>"
                        "</div>"
                    )
                    return f"❌ Внутренняя ошибка: {e}", gr.HTML.update(value=modal_html, visible=True)

                if isinstance(resp, str) and (resp.startswith("❌") or resp.startswith("⚠️") or "TypeError" in resp):
                    modal_html = (
                        "<div style='padding:16px;background:#fff;border-radius:8px;'>"
                        f"<h3 style='color:#c00;margin:0 0 8px;'>Ошибка ввода</h3>"
                        f"<p>{resp}</p>"
                        "<p>Проверьте формат и повторите попытку.</p>"
                        "</div>"
                    )
                    return resp, gr.HTML.update(value=modal_html, visible=True)

                return resp, gr.HTML.update(value="", visible=False)

            ocr_submit.click(
                fn=ocr_via_analyze_safe,
                inputs=[ocr_image, ocr_lang, ocr_max_length],
                outputs=[ocr_output, ocr_error_modal]
            )

            with gr.Row():
                ocr_clear = gr.Button("🗑️ Clear OCR")
                ocr_clear.click(
                    fn=lambda: (None, ""),
                    inputs=[],
                    outputs=[ocr_image, ocr_output]
                )
            def save_ocr_output(text):
                return _save_text_to_file(text, prefix="ocr_output")

            ocr_save_btn.click(
                fn=save_ocr_output,
                inputs=[ocr_output],
                outputs=[ocr_download]
            )

        # Settings Tab (device + vision model selector)
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Device & Vision Settings")

            # Stack settings vertically as requested
            with gr.Column():
                device_selector = gr.Dropdown(
                    choices=["auto", "cuda", "mps", "cpu"],
                    value="auto",
                    label="Device",
                    info="Select 'auto' to auto-detect CUDA/MPS/CPU"
                )

                vision_model_selector = gr.Dropdown(
                    choices=["HuggingFaceTB/SmolVLM2-2.2B-Instruct", "HuggingFaceTB/SmolVLM-256M-Instruct"],
                    value="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                    label="Vision Model",
                    info="Choose which vision model to load"
                )

                # Apply block: button with result placed under it
                with gr.Column():
                    apply_btn = gr.Button("Apply Settings", variant="primary")
                    apply_result = gr.Markdown(value="")
                    # Hidden HTML modal for errors (will be updated and made visible on error)
                    apply_error_modal = gr.HTML(visible=False)

            def _apply(device_choice, vmodel):
                # Validate device availability before applying
                if device_choice == "cuda" and not torch.cuda.is_available():
                    msg = "❌ CUDA (GPU) is not available on this machine. Settings not applied."
                    modal_html = (
                        "<div id=\"gr-modal\" style=\"position:fixed;left:0;top:0;width:100%;height:100%;"
                        "background:rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;z-index:9999;\">"
                        "<div style=\"background:#fff;padding:20px;border-radius:8px;max-width:90%;box-shadow:0 10px 30px rgba(0,0,0,0.3);\">"
                        f"<h3 style=\"margin:0 0 8px;color:#c00;\">Ошибка</h3><p>{msg}</p>"
                        "<div style=\"text-align:right;margin-top:12px;\">"
                        "<button onclick=\"document.getElementById('gr-modal').style.display='none'\">Закрыть</button>"
                        "</div></div></div>"
                    )
                    md = f"<span style='color:red'>{msg}</span>"
                    return gr.Markdown.update(value=md), gr.HTML.update(value=modal_html, visible=True)

                if device_choice == "mps":
                    mps_available = False
                    try:
                        mps_available = hasattr(torch.backends, "mps") and getattr(torch.backends.mps, "is_available", lambda: False)()
                    except Exception:
                        mps_available = False

                    if not mps_available:
                        msg = "❌ MPS (Apple Silicon) is not available on this machine. Settings not applied."
                        modal_html = (
                            "<div id=\"gr-modal\" style=\"position:fixed;left:0;top:0;width:100%;height:100%;"
                            "background:rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;z-index:9999;\">"
                            "<div style=\"background:#fff;padding:20px;border-radius:8px;max-width:90%;box-shadow:0 10px 30px rgba(0,0,0,0.3);\">"
                            f"<h3 style=\"margin:0 0 8px;color:#c00;\">Ошибка</h3><p>{msg}</p>"
                            "<div style=\"text-align:right;margin-top:12px;\">"
                            "<button onclick=\"document.getElementById('gr-modal').style.display='none'\">Закрыть</button>"
                            "</div></div></div>"
                        )
                        md = f"<span style='color:red'>{msg}</span>"
                        return gr.Markdown.update(value=md), gr.HTML.update(value=modal_html, visible=True)

                try:
                    # apply_settings will update globals (DEVICE, VISION_MODEL)
                    info = apply_settings(device_choice, vmodel)
                    success_msg = f"✅ Settings applied successfully: {info}"
                    md = f"<span style='color:green'>{success_msg}</span>"
                    # Ensure modal hidden on success
                    return gr.Markdown.update(value=md), gr.HTML.update(value="", visible=False)
                except Exception as e:
                    msg = f"❌ Failed to apply settings: {e}"
                    modal_html = (
                        "<div id=\"gr-modal\" style=\"position:fixed;left:0;top:0;width:100%;height:100%;"
                        "background:rgba(0,0,0,0.4);display:flex;align-items:center;justify-content:center;z-index:9999;\">"
                        "<div style=\"background:#fff;padding:20px;border-radius:8px;max-width:90%;box-shadow:0 10px 30px rgba(0,0,0,0.3);\">"
                        f"<h3 style=\"margin:0 0 8px;color:#c00;\">Ошибка</h3><p>{msg}</p>"
                        "<div style=\"text-align:right;margin-top:12px;\">"
                        "<button onclick=\"document.getElementById('gr-modal').style.display='none'\">Закрыть</button>"
                        "</div></div></div>"
                    )
                    md = f"<span style='color:red'>{msg}</span>"
                    return gr.Markdown.update(value=md), gr.HTML.update(value=modal_html, visible=True)

            apply_btn.click(
                fn=_apply,
                inputs=[device_selector, vision_model_selector],
                outputs=[apply_result, apply_error_modal]
            )
    

if __name__ == "__main__":
    print("\n Launching Gradio interface...")
    print("\n" + "="*60)

    demo.launch(
        server_name="0.0.0.0"
    )
    