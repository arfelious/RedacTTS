"""
Flask Web Application for RedacTTS

Routes:
- GET /           : Serve the main UI
- POST /upload    : Upload PDF and extract Q&A
- GET /voices     : Get available TTS voices
- POST /generate  : Generate TTS audio from Q&A
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import json
import time
import uuid
import threading
import generate_tts_unified

try:
    import storage
except ImportError:
    storage = None

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
FILE_MAX_AGE_SECONDS = 3600

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def cleanup_old_files():
    """Remove files older than FILE_MAX_AGE_SECONDS from upload/output folders."""
    while True:
        try:
            now = time.time()
            for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        if os.path.isfile(filepath):
                            file_age = now - os.path.getmtime(filepath)
                            if file_age > FILE_MAX_AGE_SECONDS:
                                try:
                                    os.remove(filepath)
                                    print(f"Cleanup: Removed {filepath}")
                                except Exception as e:
                                    print(f"Cleanup failed: {e}")
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(300)


cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()


VOICES = {
    "Local": [
        "p225 (Female)", "p226 (Male)", "p227 (Female)", "p228 (Male)",
        "p229 (Male)", "p230 (Male)", "p231 (Male)", "p232 (Male)", "p233 (Male)"
    ],
    "Cloud": [
        "Danielle (US Female)", "Joanna (US Female)", "Ruth (US Female)", 
        "Salli (US Female)", "Kimberly (US Female)", "Kendra (US Female)", "Ivy (US Female)",
        "Gregory (US Male)", "Kevin (US Male)", "Matthew (US Male)", 
        "Justin (US Male)", "Joey (US Male)", "Stephen (US Male)",
        "Amy (UK Female)", "Emma (UK Female)", "Brian (UK Male)", "Arthur (UK Male)",
        "Olivia (AU Female)"
    ]
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    file_data = file.read()
    if storage:
        storage.write_file(filepath, file_data)
    else:
        with open(filepath, 'wb') as f:
            f.write(file_data)
    
    ocr_lambda_name = os.environ.get('OCR_LAMBDA_NAME')
    
    if ocr_lambda_name:
        import boto3
        lambda_client = boto3.client('lambda')
        pdf_key = f"{UPLOAD_FOLDER}/{file.filename}"
        
        response = lambda_client.invoke(
            FunctionName=ocr_lambda_name,
            InvocationType='RequestResponse',
            Payload=json.dumps({'pdf_key': pdf_key})
        )
        
        result = json.loads(response['Payload'].read())
        if result.get('statusCode') != 200:
            error_body = json.loads(result.get('body', '{}'))
            raise Exception(error_body.get('error', 'Lambda OCR failed'))
        
        body = json.loads(result['body'])
        json_path = body['qa_json_key']
        qa_data = json.loads(storage.read_text(json_path))
    else:
        extracted_text_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                           file.filename.replace('.pdf', '_extracted.txt'))
        subprocess.run(['.\\venv\\Scripts\\python', 'pdf_redaction_extractor.py', filepath, 
                       '--output', app.config['UPLOAD_FOLDER']], check=True)
        subprocess.run(['.\\venv\\Scripts\\python', 'extract_qa.py', extracted_text_path], check=True)
        
        json_path = extracted_text_path.replace('.txt', '_qa.json')
        
        if storage:
            qa_data = json.loads(storage.read_text(json_path))
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
        
    return jsonify({'qa_pairs': qa_data, 'filename': file.filename})


@app.route('/voices', methods=['GET'])
def get_voices():
    return jsonify(VOICES)


@app.route('/generate', methods=['POST'])
def generate_tts():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename missing'}), 400
        
    json_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                             filename.replace('.pdf', '_extracted_qa.json'))
    
    if storage:
        if not storage.file_exists(json_path):
            return jsonify({'error': 'Source data not found'}), 404
        qa_pairs = json.loads(storage.read_text(json_path))
    else:
        if not os.path.exists(json_path):
            return jsonify({'error': 'Source data not found'}), 404
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)

    count = int(data.get('count', 30))
    voice_q = data.get('voice_q', 'Matthew (US Male)')
    voice_a = data.get('voice_a', 'Joanna (US Female)')
    voice_extra = data.get('voice_extra', 'Brian (UK Male)')
    
    mode = "Local" if voice_q.startswith("p") and "(" in voice_q else "Cloud"
    
    def clean_voice(v):
        return v.split(" (")[0] if v else ""
    
    v_q = clean_voice(voice_q)
    v_a = clean_voice(voice_a)
    v_e = clean_voice(voice_extra)
    
    ext = "wav" if mode == "Local" else "mp3"
    output_filename = f"tts_{uuid.uuid4()}.{ext}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    
    if count > 0:
        qa_pairs = qa_pairs[:count]
    
    # Redaction audio: silence, pink, or white noise
    redaction_audio = data.get('redaction_audio', 'silence')
    noise_path = None
    if redaction_audio == 'pink':
        noise_path = 'static/pink.mp3'
    elif redaction_audio == 'white':
        noise_path = 'static/white.mp3'

    try:
        if mode == "Local":
            generate_tts_unified.generate_coqui(qa_pairs, v_q, v_a, v_e, output_path, noise_path)
        else:
            generate_tts_unified.generate_polly(qa_pairs, v_q, v_a, v_e, output_path, noise_path)
    except Exception as e:
        return jsonify({'error': f'TTS failed: {str(e)}'}), 500

    mime = 'audio/wav' if mode == 'Local' else 'audio/mpeg'
    return send_file(os.path.abspath(output_path), mimetype=mime, as_attachment=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host='127.0.0.1', port=args.port, debug=True)
