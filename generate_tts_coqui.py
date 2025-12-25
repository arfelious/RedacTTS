import argparse
import json
import os
import numpy as np
from scipy.io.wavfile import write
import torch

# Try extracting TTS, catching import errors if not installed yet
try:
    from TTS.api import TTS
except ImportError:
    print("Error: Coqui TTS not installed. Run 'pip install TTS'")
    exit(1)

def generate_audio(json_path, count, output_path, model_name="tts_models/en/vctk/vits"):
    print(f"Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Slice first N
    if count > 0:
        data = data[:count]
    
    print(f"Initializing TTS with model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name=model_name, progress_bar=False).to(device)
    
    # Check speakers if multi-speaker
    speakers = tts.speakers if tts.is_multi_speaker else []
    print(f"Model Speakers: {len(speakers)}")
    
    # Assign speakers (simple heuristic for VCTK)
    # VCTK has IDs liek 'p225', 'p226', etc.
    speaker_q = speakers[0] if speakers else None
    speaker_a = speakers[1] if len(speakers) > 1 else speaker_q
    speaker_e = speakers[2] if len(speakers) > 2 else speaker_q
    
    print(f"Voice Assignments:\n  Q: {speaker_q}\n  A: {speaker_a}\n  Extra: {speaker_e}")
    
    all_audio = []
    sample_rate = tts.synthesizer.output_sample_rate
    
    print("Generating audio...")
    for i, item in enumerate(data):
        print(f"Processing Item {i+1}/{len(data)}")
        
        # Helper to generate and append
        def add_segment(text, speaker):
            if not text: return
            # Clean text slightly?
            clean_text = text.replace('\n', ' ').strip()
            if not clean_text: return
            
            # Generate
            # TTS api returns a list of floats
            try:
                wav = tts.tts(text=clean_text, speaker=speaker)
                all_audio.extend(wav)
                
                # Add silence (0.5s)
                silence_len = int(sample_rate * 0.5)
                all_audio.extend([0.0] * silence_len)
            except Exception as e:
                print(f"  Error generating segment: {e}")

        # Extract Question
        q_text = item.get('Question', {}).get('text', "")
        if q_text:
            add_segment(q_text, speaker_q)
            
        # Extract Answer
        a_text = item.get('Answer', {}).get('text', "")
        if a_text:
            add_segment(a_text, speaker_a)
            
        # Extract Extra
        e_text = item.get('Extra', {}).get('text', "")
        if e_text:
            e_name = item.get('Extra', {}).get('name', "Info")
            # Maybe announce speaker? "Grand Juror says..."
            # For now just read text
            add_segment(e_text, speaker_e)
            
    # Convert to numpy and save
    print(f"Saving to {output_path}...")
    # Scale float to int16 for wavfile.write? 
    # TTS output is usually -1.0 to 1.0 floats.
    # scipy.io.wavfile.write handles floats effectively (saves as 32-bit float usually) 
    # or we can convert to 16-bit PCM for compatibility.
    
    audio_np = np.array(all_audio, dtype=np.float32)
    
    # Normalize?
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val
        
    # Convert to 16-bit PCM
    audio_int16 = (audio_np * 32767).astype(np.int16)
    
    write(output_path, sample_rate, audio_int16)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS from extracted Q&A JSON")
    parser.add_argument("json_path", help="Path to input JSON file")
    parser.add_argument("--count", type=int, default=30, help="Number of Q/A pairs to process")
    parser.add_argument("--output", default="output.wav", help="Output WAV file path")
    parser.add_argument("--model", default="tts_models/en/vctk/vits", help="Coqui TTS model name")
    
    args = parser.parse_args()
    
    generate_audio(args.json_path, args.count, args.output, args.model)
