# RedacTTS

TTS for redacted legal transcripts.

## Setup

```bash
pip install -r requirements.txt
python app.py -p 8080  # or default 5000
```

Open http://localhost:5000

## Config

Set via environment variables:

- `PORT` - Server port (default: 5000)
- `STORAGE_BACKEND` - `local` or `s3`
- `S3_BUCKET` - Bucket name for S3 mode
- `POLLY_ROLE_ARN` - IAM role for Polly
- `OCR_LAMBDA_NAME` - Lambda function for OCR

## Files

- `app.py` - Flask server
- `pdf_redaction_extractor.py` - PDF text extraction
- `extract_qa.py` - Q&A parsing
- `generate_tts_unified.py` - TTS (Coqui/Polly)
- `storage.py` - Local/S3 abstraction
- `lambda_handler.py` - AWS Lambda handler
