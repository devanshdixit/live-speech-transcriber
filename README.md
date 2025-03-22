# 🎙️ Live Speech Transcriber

A real-time speech transcription tool built with **Streamlit**, leveraging **Whisper from Hugging Face** to transcribe long audio files accurately. This app also provides pronunciation feedback by comparing the transcribed output with the sentence the user intended to say.

## 🚀 Features

- ✅ Supports long audio uploads (WAV, MP3, M4A, FLAC, etc.)
- 🧠 Automatically splits and transcribes long recordings into 30s chunks
- 📊 Computes **Pronunciation Accuracy** using Word Error Rate (WER)
- 🔍 Highlights the most similar transcribed segment to user's intended sentence
- 📦 Uses Whisper (base) model from Hugging Face
- 📤 Clean and simple UI using Streamlit

## 🧪 Usage

1. Clone the repository:

```bash
git clone https://github.com/devanshdixit/live-speech-transcriber.git
cd live-speech-transcriber
```

2. Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – frontend and backend integration
- **Whisper (Hugging Face Transformers)** – transcription
- **Librosa, Pydub, Torchaudio** – audio processing
- **Jiwer** – pronunciation scoring with WER

## 📂 Folder Structure

```
live-speech-transcriber/
├── app.py              # Main app file (Streamlit UI + Whisper integration)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 💡 Example Use Case

Ideal for language learners, public speakers, or developers building tools for:
- Live transcription in meetings
- Accent reduction training
- Pronunciation scoring in language learning apps

## 📈 Roadmap

- ⏺️ Integrate real-time browser mic input
- 🪄 Deploy model inference to GPU backend for speedup
- 📥 Upload transcribed chunks to backend for post-processing
- 📋 Export meeting transcripts

## 👨‍💻 Author

**Devanshu Dixit**  
Data Scientist | AI Engineer | Full-Stack Developer  
📫 [LinkedIn](https://linkedin.com/in/devanshu-dixit)  
✉️ devanshudixit@vt.edu

---

🎧 *Try speaking a technical sentence and check your clarity and fluency!*