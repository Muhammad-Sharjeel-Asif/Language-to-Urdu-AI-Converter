import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
import torch

LANGUAGE_MODELS = {
    "English": "Helsinki-NLP/opus-mt-en-ur",
    "Hindi": "Helsinki-NLP/opus-mt-hi-ur",
    "Arabic": "Helsinki-NLP/opus-mt-ar-en",
    "French": "Helsinki-NLP/opus-mt-fr-en",
    "Spanish": "Helsinki-NLP/opus-mt-es-en",
    "German": "Helsinki-NLP/opus-mt-de-en",
    "Turkish": "Helsinki-NLP/opus-mt-tr-en",
    "Persian": "Helsinki-NLP/opus-mt-fa-en",
    "Chinese (Simplified)": "Helsinki-NLP/opus-mt-zh-en",
    "Russian": "Helsinki-NLP/opus-mt-ru-en"
}

EN_TO_UR_MODEL = "Helsinki-NLP/opus-mt-en-ur"

model_cache = {}

def load_model(model_name):
    """
    Load and cache models to avoid reloading every time.
    """
    if model_name not in model_cache:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model_cache[model_name] = (tokenizer, model)
    return model_cache[model_name]

def translate_text(text, source_lang):
    """
    Handles translation logic:
    1. Source -> English (if needed)
    2. English -> Urdu
    """
    if not text.strip():
        return "Please enter some text to translate."

    try:
        # Get model for source language
        src_model_name = LANGUAGE_MODELS[source_lang]

        # If direct English -> Urdu
        if source_lang == "English":
            tokenizer, model = load_model(src_model_name)
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)

        # If direct Hindi -> Urdu
        if source_lang == "Hindi":
            tokenizer, model = load_model(src_model_name)
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)

        # Step 1: Source -> English
        tokenizer1, model1 = load_model(src_model_name)
        inputs1 = tokenizer1(text, return_tensors="pt", padding=True)
        translated1 = model1.generate(**inputs1)
        english_text = tokenizer1.decode(translated1[0], skip_special_tokens=True)

        # Step 2: English -> Urdu
        tokenizer2, model2 = load_model(EN_TO_UR_MODEL)
        inputs2 = tokenizer2(english_text, return_tensors="pt", padding=True)
        translated2 = model2.generate(**inputs2)
        urdu_text = tokenizer2.decode(translated2[0], skip_special_tokens=True)

        return urdu_text

    except Exception as e:
        return f"Error during translation: {str(e)}"

def count_chars(text):
    return f"Characters: {len(text)}"

custom_css = """
#urdu_output textarea {
    direction: rtl;
    text-align: right;
    font-size: 18px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:

    gr.Markdown("# 🌐 Multilingual to Urdu Translator")
    gr.Markdown("Translate text from multiple languages into Urdu using AI-powered models.")

    with gr.Row():
        # Left Column 
        with gr.Column():
            source_lang = gr.Dropdown(
                choices=list(LANGUAGE_MODELS.keys()),
                value="English",
                label="Select Source Language"
            )

            input_text = gr.Textbox(
                lines=8,
                placeholder="Enter text here...",
                label="Input Text"
            )

            char_count = gr.Markdown("Characters: 0")

            with gr.Row():
                translate_btn = gr.Button("Translate")
                clear_btn = gr.Button("Clear")

        # Right Column (Output)
        with gr.Column():
            output_text = gr.Textbox(
                lines=8,
                label="Urdu Translation",
                interactive=False,
                elem_id="urdu_output"
            )

    gr.Examples(
        examples=[
            ["Hello, how are you?", "English"],
            ["नमस्ते, आप कैसे हैं?", "Hindi"],
            ["Bonjour tout le monde", "French"],
            ["Hola amigo", "Spanish"],
            ["مرحبا كيف حالك", "Arabic"]
        ],
        inputs=[input_text, source_lang]
    )

    
    input_text.change(fn=count_chars, inputs=input_text, outputs=char_count)

    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, source_lang],
        outputs=output_text
    )

    def clear_fields():
        return "", "", "Characters: 0"

    clear_btn.click(
        fn=clear_fields,
        outputs=[input_text, output_text, char_count]
    )


if __name__ == "__main__":
    demo.launch()