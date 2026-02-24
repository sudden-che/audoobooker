import asyncio
from pathlib import Path

# mock
EDGE_VOICES = ["edge1", "edge2"]
SILERO_SPEAKERS = ["sil1", "sil2"]

def split_text(text, size):
    return [text]

def generate_audio(input_data, settings):
    import random
    chunk_size = settings.get("chunk_size", 10000)
    engine = settings.get("engine", "edge")

    if settings.get("random"):
        engine = random.choice(["edge", "silero"])

    if engine == "silero" and chunk_size > 800:
        chunk_size = 800
    
    if settings.get("random") and engine == "silero":
        settings["silero_model_id"] = "v5_ru"

    sender_voices = {}
    tasks_data = [] 
    
    if isinstance(input_data, str):
        assigned_voice = None
        if settings.get("random"):
            if engine == "edge":
                assigned_voice = random.choice(EDGE_VOICES)
            else:
                assigned_voice = random.choice(SILERO_SPEAKERS)
        
        chunks = split_text(input_data, chunk_size)
        for c in chunks:
            tasks_data.append((c, assigned_voice))
    else:
        for text_part, sender_id in input_data:
            p_chunks = split_text(text_part, chunk_size)
            
            assigned_voice = None
            if settings.get("random"):
                if sender_id not in sender_voices:
                    if engine == "edge":
                        assigned_voice = random.choice(EDGE_VOICES)
                    else:
                        assigned_voice = random.choice(SILERO_SPEAKERS)
                    sender_voices[sender_id] = assigned_voice
                else:
                    assigned_voice = sender_voices[sender_id]
            
            for pc in p_chunks:
                tasks_data.append((pc, assigned_voice))

    print(f"Tasks: {tasks_data}")

input_data = [("msg1", 123), ("msg2", 456)]
settings = {"random": True}
for _ in range(5):
    generate_audio(input_data, settings.copy())

