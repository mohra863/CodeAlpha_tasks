import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import base64

st.set_page_config(
    page_title="Language Translation Tool",
    page_icon="🌍",
    layout="centered"
)

st.title("Language Translation Tool")

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("pic2.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
text = st.text_area("Enter the text you want to translate:")

languages = GoogleTranslator().get_supported_languages(as_dict=True)
lang_names = list(languages.keys())

source_lang = st.selectbox("Source Language:", ["auto"] + lang_names)
target_lang = st.selectbox("Target Language:", lang_names)

if "translated" not in st.session_state:
    st.session_state.translated = ""

if st.button("Translate"):
    if text.strip() == "":
        st.warning("please enter your text")
    else:
        try:
            translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
            st.session_state.translated = translated
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.session_state.translated:
    st.success("Translation completed successfully")
    st.code(st.session_state.translated, language=None)

    if st.button("🔊 listen"):
        try:
            lang_code = languages[target_lang]
            tts = gTTS(text=st.session_state.translated, lang=lang_code)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error("Unable to generate audio for the selected language: {e}")