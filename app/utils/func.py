import wave
from dotenv import load_dotenv
import os
import torch
from openai import OpenAI
from vad import EnergyVAD
from pydub import AudioSegment
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf

load_dotenv()
client = OpenAI()

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
vad_iterator = VADIterator(model)

def answer_question(msg: str):    
    instructor = f"""
        You will act as a kind assistant.
        Please answer within 180 characters.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {'role': "system", "content": instructor},
                {'role': "user", "content": msg}    
            ],
            max_tokens=3000,
            # stream=True
        )
        print("response: ", response)
        return response.choices[0].message.content        
    except Exception as e:
        print(e)

def load_audio(audio_path, sample_rate=16000):
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return waveform

def detect_speech(audio_path):
    # audio = load_audio(audio_path)
    # vad = EnergyVAD(frame_length=1000, frame_shift=500, energy_threshold=0.3)
    # return vad(waveform=audio)
    wav = read_audio(audio_path, sampling_rate=16000)
    window_size_samples = 16000 # number of samples in a single audio chunk
    speech_dict = []
    for i in range(0, len(wav), window_size_samples * 2):
        chunk = wav[i: i+ window_size_samples * 2]
        if len(chunk) < window_size_samples:
            break
        speech_dict.append(vad_iterator(chunk, return_seconds=True))
    print(speech_dict)
    return not all(item is None for item in speech_dict)

def transcribe_audio(filename):
    audio_file_path = f"data/{filename}"
    y, sr = librosa.load(audio_file_path, sr=None)  # y is the audio time series, sr is the sample rate

    # Perform noise reduction
    y_reduced = nr.reduce_noise(y=y, sr=sr)

    # Save the cleaned audio
    cleaned_file_path = f"data/cleaned_{filename}"
    sf.write(cleaned_file_path, y_reduced, sr)

    audio_file= open(cleaned_file_path, "rb")
    
    if detect_speech(cleaned_file_path):
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            # language="az"
        )
        print(transcription.text)
        return transcription.text
        
    else:
        print("No speech detected in the audio file.")
        return "No speech"
        