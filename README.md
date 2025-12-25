# RedacTTS

Text-to-speech generation for redacted legal transcripts. Extracts Q&A pairs from PDFs with redaction detection and generates narrated audio. The project uses Coqui and Amazon Polly for TTS generation and allows specifying different neural voices for questions, answers, and extras. (speakers, grand juror, etc.).


## Configuration

Most of the configuration is done via environment variables and are optional, local mode will be used if environment variables are not provided.

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

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py -p 8080  # or default 5000


# Open http://localhost:<port>
```
## Lambda Deployment
See [LAMBDA_DEPLOY.md](LAMBDA_DEPLOY.md) for container deployment instructions.

## License

MIT


