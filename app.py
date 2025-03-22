import torch
import streamlit as st
import os
import time
import torchaudio
import librosa
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# 🔧 Fix torch.classes issue in Streamlit
import sys
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# 🔄 Cache Whisper model
@st.cache_resource
def load_model():
    model_id = "openai/whisper-base"  # Try "small", "medium", or "large-v3" for better accuracy
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    return processor, model

processor, model = load_model()

st.title("🎵 Pronunciation Practice & Long Audio Transcriber (Powered by Whisper)")
st.write("Upload a long or short audio file and get your transcription instantly.")

SUPPORTED_FORMATS = ["wav", "aiff", "m4a", "mp3", "flac"]
audio_file = st.file_uploader("Upload an audio file", type=SUPPORTED_FORMATS)


def transcribe_long_audio(file_path, chunk_duration=30):
    audio, sr = librosa.load(file_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    full_transcript = ""
    total_chunks = int(total_duration) // chunk_duration + 1
    st.info(f"🔍 Splitting audio into {total_chunks} chunk(s) of {chunk_duration} seconds each")

    for i, start in enumerate(range(0, int(total_duration), chunk_duration)):
        end = start + chunk_duration
        chunk = audio[start * sr:end * sr]

        if len(chunk) < 1000:
            st.warning(f"⏩ Skipping chunk {i+1} (too short)")
            continue

        st.write(f"📦 Processing chunk {i+1}: {start}–{min(end, int(total_duration))}s")

        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_features = inputs["input_features"]

        current_len = input_features.shape[-1]
        if current_len < 3000:
            pad_width = 3000 - current_len
            input_features = torch.nn.functional.pad(input_features, (0, pad_width), mode="constant", value=0.0)

        attention_mask = torch.ones_like(input_features[:, 0, :])
        inputs["input_features"] = input_features
        inputs["attention_mask"] = attention_mask

        with torch.no_grad():
            ids = model.generate(inputs["input_features"])
        chunk_text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        st.write(f"📝 Chunk {i+1} transcription: {chunk_text}")
        full_transcript += chunk_text + " "

    return full_transcript.strip()

if audio_file:
    file_extension = audio_file.name.split(".")[-1].lower()
    file_path = f"temp_input.{file_extension}"
    wav_path = "temp_converted.wav"

    # Save file locally
    with open(file_path, "wb") as f:
        f.write(audio_file.read())

    # Convert to 16kHz mono WAV using PyDub
    audio = AudioSegment.from_file(file_path, format=file_extension)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")

    with st.spinner("🧠 Transcribing long audio with Whisper model..."):
        t0 = time.time()
        transcription = transcribe_long_audio(wav_path)
        t1 = time.time()

    st.success(f"✅ Transcription complete in {round(t1 - t0, 2)} seconds")
    st.subheader("🖋️ Transcribed Text")
    st.write(transcription)
    # Input target/reference sentence from user
    reference_text = st.text_input("📌 Enter the sentence you intended to say:", value=transcription)
    # Pronunciation Scoring
    if reference_text and reference_text.strip() != "":
        st.subheader("📊 Pronunciation Feedback")
        st.caption("Comparing transcription with the expected sentence:")
        st.code(reference_text)

        error_rate = wer(reference_text.lower(), transcription.lower())
        score = max(0, 1 - error_rate)
        st.metric("🗣️ Pronunciation Accuracy", f"{score * 100:.2f}%")

        # Extract most similar segment
        best_match = ""
        min_wer = 1.0
        for sent in transcription.split("."):
            sent = sent.strip()
            if not sent:
                continue
            cur_wer = wer(reference_text.lower(), sent.lower())
            if cur_wer < min_wer:
                min_wer = cur_wer
                best_match = sent

        if best_match:
            st.caption("🔍 Matched Transcribed Segment:")
            st.write(best_match)

        if score >= 0.9:
            st.success("💚 Excellent pronunciation!")
        elif score >= 0.7:
            st.warning("💛 Good, but could be clearer")
        else:
            st.error("❤️ Needs improvement — try speaking slowly and clearly")

    # Cleanup
    os.remove(file_path)
    os.remove(wav_path)

st.write("🎧 Try speaking a technical sentence and check your clarity and fluency!")
