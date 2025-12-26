"""
Unified TTS Generation Module

Supports two backends:
- Local: Coqui TTS (VCTK/VITS model)
- Cloud: Amazon Polly (Neural voices)

Handles [REDACTED n] markers by inserting proportional silence or white noise.
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


def load_white_noise_wav(white_noise_path, target_sample_rate):
    """Load white noise MP3 and convert to WAV samples at target sample rate."""
    from pydub import AudioSegment
    import numpy as np
    
    noise = AudioSegment.from_mp3(white_noise_path)
    # Convert to target sample rate
    noise = noise.set_frame_rate(target_sample_rate).set_channels(1)
    # Convert to numpy array (normalized float)
    samples = np.array(noise.get_array_of_samples(), dtype=np.float32)
    samples = samples / 32768.0  # Normalize to -1.0 to 1.0
    return samples


def generate_coqui(data, voice_q, voice_a, voice_e, output_path, white_noise_path=None):
    """Generate audio using Coqui TTS (local neural TTS)."""
    import numpy as np
    from scipy.io.wavfile import write
    
    ensure_espeak_path()
    tts = get_coqui_model()
    sample_rate = tts.synthesizer.output_sample_rate
    
    # Load white noise samples if provided
    white_noise_samples = None
    if white_noise_path and os.path.exists(white_noise_path):
        try:
            white_noise_samples = load_white_noise_wav(white_noise_path, sample_rate)
            print(f"Loaded white noise: {len(white_noise_samples)} samples")
        except Exception as e:
            print(f"Failed to load white noise: {e}")
    
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
                    num_samples = int(duration * sample_rate)
                    
                    if white_noise_samples is not None:
                        # Use white noise - loop/trim to desired length
                        noise_len = len(white_noise_samples)
                        if noise_len >= num_samples:
                            all_audio.extend(white_noise_samples[:num_samples])
                        else:
                            # Loop the noise
                            loops = num_samples // noise_len + 1
                            looped = np.tile(white_noise_samples, loops)
                            all_audio.extend(looped[:num_samples])
                    else:
                        # Use silence
                        all_audio.extend([0.0] * num_samples)
                else:
                    clean_text = part.replace('\n', ' ').strip()
                    if clean_text:
                        # Split by -- for pauses
                        dash_parts = re.split(r'--+', clean_text)
                        for j, dash_part in enumerate(dash_parts):
                            dash_part = dash_part.strip()
                            if dash_part:
                                try:
                                    wav = tts.tts(text=dash_part, speaker=speaker)
                                    all_audio.extend(wav)
                                except Exception as e:
                                    print(f"  Error: {e}")
                            # Add pause between parts (except after last)
                            if j < len(dash_parts) - 1:
                                all_audio.extend([0.0] * int(sample_rate * 0.2))
                        all_audio.extend([0.0] * int(sample_rate * 0.1))

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


def load_white_noise_chunk(white_noise_path, duration_sec):
    """Load and trim/loop white noise to desired duration."""
    from pydub import AudioSegment
    
    noise = AudioSegment.from_mp3(white_noise_path)
    target_ms = int(duration_sec * 1000)
    
    # Loop if too short
    while len(noise) < target_ms:
        noise = noise + noise
    
    # Trim to exact length
    return noise[:target_ms]


def generate_polly(data, voice_q, voice_a, voice_e, output_path, white_noise_path=None):
    """Generate audio using Amazon Polly (cloud neural TTS).
    
    Args:
        white_noise_path: Optional path to white noise MP3. If provided, uses
                         white noise for redacted sections instead of silence.
    """
    import boto3
    from io import BytesIO
    from contextlib import closing
    from pydub import AudioSegment
    
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
    
    audio_segments = []
    SECONDS_PER_UNIT = 0.08
    
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}...")
        
        def process_text(text, voice_id):
            if not text:
                return
            
            # Split text by redaction markers
            parts = re.split(r'(\[REDACTED\s+\d+\])', text)
            
            for part in parts:
                if not part.strip():
                    continue
                
                redaction_match = re.match(r'\[REDACTED\s+(\d+)\]', part)
                if redaction_match:
                    num_units = int(redaction_match.group(1))
                    duration_sec = min(10.0, max(0.1, num_units * SECONDS_PER_UNIT))
                    
                    if white_noise_path and os.path.exists(white_noise_path):
                        # Use white noise
                        noise_chunk = load_white_noise_chunk(white_noise_path, duration_sec)
                        audio_segments.append(noise_chunk)
                    else:
                        # Use silence
                        silence = AudioSegment.silent(duration=int(duration_sec * 1000))
                        audio_segments.append(silence)
                else:
                    # Synthesize speech
                    clean_text = part.replace('\n', ' ').strip()
                    if clean_text:
                        safe_text = escape(clean_text)
                        # Replace -- with small break
                        safe_text = re.sub(r'--+', '<break time="200ms"/>', safe_text)
                        try:
                            response = polly.synthesize_speech(
                                Text=f"<speak>{safe_text}</speak>",
                                TextType='ssml',
                                OutputFormat='mp3',
                                VoiceId=voice_id,
                                Engine='neural'
                            )
                            if "AudioStream" in response:
                                with closing(response["AudioStream"]) as stream:
                                    audio_data = stream.read()
                                segment = AudioSegment.from_mp3(BytesIO(audio_data))
                                audio_segments.append(segment)
                        except Exception as e:
                            print(f"Polly Error ({voice_id}): {e}")
            
            # Add small gap after each text block
            audio_segments.append(AudioSegment.silent(duration=100))
        
        process_text(item.get('Question', {}).get('text'), voice_q)
        process_text(item.get('Answer', {}).get('text'), voice_a)
        process_text(item.get('Extra', {}).get('text'), voice_e)
        
        # Gap between items
        audio_segments.append(AudioSegment.silent(duration=500))

    if audio_segments:
        if not output_path.lower().endswith('.mp3'):
            output_path += '.mp3'
        
        combined = sum(audio_segments, AudioSegment.empty())
        combined.export(output_path, format="mp3")
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
    parser.add_argument("--white-noise", default=None, help="Path to white noise MP3 for redactions")
    
    args = parser.parse_args()
    
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.count > 0:
        data = data[:args.count]
        
    if args.mode == "Local":
        generate_coqui(data, args.voice_q, args.voice_a, args.voice_e, args.output)
    else:
        generate_polly(data, args.voice_q, args.voice_a, args.voice_e, args.output, args.white_noise)


if __name__ == "__main__":
    # Import BytesIO for CLI usage
    from io import BytesIO
    main()
