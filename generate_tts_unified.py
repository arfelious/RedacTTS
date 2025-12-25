"""
Unified TTS Generation Module

Supports two backends:
- Local: Coqui TTS (VCTK/VITS model)
- Cloud: Amazon Polly (Neural voices)

Handles [REDACTED n] markers by inserting proportional silence.
"""

import os
import sys
import json
import re
import argparse
from xml.sax.saxutils import escape

_COQUI_TTS = None
_DEVICE = None


def ensure_espeak_path():
    """Configure eSpeak path for Windows if not already set."""
    if os.environ.get('PHONEMIZER_ESPEAK_PATH'):
        return
    
    possible_paths = [
        r"C:\Program Files\eSpeak NG",
        r"C:\Program Files (x86)\eSpeak NG"
    ]
    for p in possible_paths:
        if os.path.exists(p) and p not in os.environ['PATH']:
            os.environ['PATH'] += ";" + p
            os.environ['PHONEMIZER_ESPEAK_PATH'] = os.path.join(p, 'espeak-ng.exe')
            break


def get_coqui_model():
    """Lazy load and cache the Coqui TTS model."""
    global _COQUI_TTS, _DEVICE
    
    if _COQUI_TTS is None:
        try:
            import torch
            from TTS.api import TTS
        except ImportError:
            print("Error: Coqui TTS not installed.")
            sys.exit(1)
            
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Coqui TTS (device: {_DEVICE})...")
        _COQUI_TTS = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False).to(_DEVICE)
        
    return _COQUI_TTS


def generate_coqui(data, voice_q, voice_a, voice_e, output_path):
    """Generate audio using Coqui TTS (local neural TTS)."""
    import numpy as np
    from scipy.io.wavfile import write
    
    ensure_espeak_path()
    tts = get_coqui_model()
    sample_rate = tts.synthesizer.output_sample_rate
    
    all_audio = []
    SECONDS_PER_UNIT = 0.08
    
    print("Generating audio (Local)...")
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}")
        
        def process_segment(text, speaker):
            if not text:
                return
            
            parts = re.split(r'(\[REDACTED\s+\d+\])', text)
            
            for part in parts:
                if not part.strip():
                    continue
                
                redaction_match = re.match(r'\[REDACTED\s+(\d+)\]', part)
                if redaction_match:
                    num_units = int(redaction_match.group(1))
                    duration = min(10.0, max(0.1, num_units * SECONDS_PER_UNIT))
                    all_audio.extend([0.0] * int(duration * sample_rate))
                else:
                    clean_text = part.replace('\n', ' ').strip()
                    if clean_text:
                        try:
                            wav = tts.tts(text=clean_text, speaker=speaker)
                            all_audio.extend(wav)
                            all_audio.extend([0.0] * int(sample_rate * 0.1))
                        except Exception as e:
                            print(f"  Error: {e}")

            all_audio.extend([0.0] * int(sample_rate * 0.5))

        process_segment(item.get('Question', {}).get('text'), voice_q)
        process_segment(item.get('Answer', {}).get('text'), voice_a)
        process_segment(item.get('Extra', {}).get('text'), voice_e)

    if all_audio:
        audio_np = np.clip(np.array(all_audio, dtype=np.float32), -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        if not output_path.lower().endswith('.wav'):
            output_path += '.wav'
        write(output_path, sample_rate, audio_int16)
        print(f"Saved to {output_path}")
    else:
        raise Exception("No audio generated")


def generate_polly(data, voice_q, voice_a, voice_e, output_path):
    """Generate audio using Amazon Polly (cloud neural TTS)."""
    import boto3
    from contextlib import closing
    
    print("Initializing Amazon Polly...")
    
    role_arn = os.environ.get('POLLY_ROLE_ARN')
    if role_arn:
        sts = boto3.client('sts')
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName='RedacTTSSession')['Credentials']
        polly = boto3.client(
            'polly',
            aws_access_key_id=creds['AccessKeyId'],
            aws_secret_access_key=creds['SecretAccessKey'],
            aws_session_token=creds['SessionToken']
        )
    else:
        polly = boto3.client('polly')
    
    temp_files = []
    SECONDS_PER_UNIT = 0.08
    
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}...")
        
        def process_ssml(text, voice_id, suffix):
            if not text:
                return
            
            safe_text = escape(text)
            
            def repl(match):
                n = int(match.group(1))
                ms = min(10000, max(0, int(n * SECONDS_PER_UNIT * 1000)))
                return f'<break time="{ms}ms"/>'
            
            ssml_text = re.sub(r'\[REDACTED\s+(\d+)\]', repl, safe_text)
            
            try:
                response = polly.synthesize_speech(
                    Text=f"<speak>{ssml_text}</speak>",
                    TextType='ssml',
                    OutputFormat='mp3',
                    VoiceId=voice_id,
                    Engine='neural'
                )
            except Exception as e:
                print(f"Polly Error ({voice_id}): {e}")
                return

            if "AudioStream" in response:
                fname = f"temp_polly_{i}_{suffix}.mp3"
                with closing(response["AudioStream"]) as stream:
                    with open(fname, "wb") as f:
                        f.write(stream.read())
                temp_files.append(fname)
        
        process_ssml(item.get('Question', {}).get('text'), voice_q, 'q')
        process_ssml(item.get('Answer', {}).get('text'), voice_a, 'a')
        process_ssml(item.get('Extra', {}).get('text'), voice_e, 'e')

    if temp_files:
        if not output_path.lower().endswith('.mp3'):
            output_path += '.mp3'
            
        with open(output_path, 'wb') as outfile:
            for fname in temp_files:
                if os.path.exists(fname):
                    with open(fname, 'rb') as infile:
                        outfile.write(infile.read())
                    try:
                        os.remove(fname)
                    except:
                        pass
        print(f"Saved to {output_path}")
    else:
        raise Exception("No audio generated (Polly)")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio from Q&A JSON")
    parser.add_argument("json_path", help="Path to Q&A JSON file")
    parser.add_argument("--mode", choices=['Local', 'Cloud'], required=True)
    parser.add_argument("--count", type=int, default=30, help="Max items to process")
    parser.add_argument("--output", default="output", help="Output file path")
    parser.add_argument("--voice_q", default="", help="Voice for Questions")
    parser.add_argument("--voice_a", default="", help="Voice for Answers")
    parser.add_argument("--voice_e", default="", help="Voice for Extra")
    
    args = parser.parse_args()
    
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.count > 0:
        data = data[:args.count]
        
    if args.mode == "Local":
        generate_coqui(data, args.voice_q, args.voice_a, args.voice_e, args.output)
    else:
        generate_polly(data, args.voice_q, args.voice_a, args.voice_e, args.output)


if __name__ == "__main__":
    main()
