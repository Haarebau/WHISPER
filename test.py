import pyaudio
import numpy as np
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM, pipeline
import threading
import queue


q = queue.Queue()
torch.cuda.empty_cache()

def recording():
    RATE = 16000  # 16kHz
    CHUNK = 48000
    FORMAT = pyaudio.paInt16 

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("start to talk...")
    try:
        while True:
            audio_data = stream.read(CHUNK)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_np.astype(np.float32) / 32768.0
            q.put(audio_float32)       
    except KeyboardInterrupt:
        print("Stopped")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def processing():
    torch.cuda.empty_cache()# GPU
    Audio2TXT = whisper.load_model("turbo")
    model_path = "nllb-600m"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
    tgt_language = "zho_Hans" 
    translator=pipeline('translation',
                     model=nllb,
                     tokenizer=tokenizer,
                     src_language='deu_Latn',
                     tgt_language=tgt_language,
                     max_length=512
    )

    while True:
        audio = q.get()
        result = Audio2TXT.transcribe(audio, fp16=False)
        if result['text'].strip():
                translated_text=translator(result['text'])
                print(f": {result['text']}\n{translated_text[0]['translation_text']}")


t1 = threading.Thread(target=recording)
t2 = threading.Thread(traget=processing)
t1.start()
t2.start()

t1.join()
q.join()