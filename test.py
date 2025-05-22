import pyaudio
import numpy as np
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM, pipeline

torch.cuda.empty_cache()

Audio2TXT = whisper.load_model("turbo")
model_path = "nllb-600m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
nllb = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
tgt_lang = "zho_Hans"    # 简体中文

RATE = 16000  # 16kHz
CHUNK = 48000
FORMAT = pyaudio.paInt16 

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

translator=pipeline('translation',
                     model=nllb,
                     tokenizer=tokenizer,
                     src_lang='deu_Latn',
                     tgt_lang='zho_Hans',
                     max_length=512
)

print("start to talk...")

try:
    while True:
        audio_data = stream.read(CHUNK)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_np.astype(np.float32) / 32768.0
        result = Audio2TXT.transcribe(audio_float32, fp16=False)
        if result['text'].strip():
            translated_text=translator(result['text'])
            print(f": {result['text']}\n{translated_text[0]['translation_text']}")
        
except KeyboardInterrupt:
    print("Stopped")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()