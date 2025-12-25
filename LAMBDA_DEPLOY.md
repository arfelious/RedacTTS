# Lambda OCR Deployment Guide

## Prerequisites
- AWS CLI configured
- Docker
- Python 3.11+

### Build & Deploy
```bash
# Build image
docker build -t redactts-ocr .

# Create ECR repo
aws ecr create-repository --repository-name redactts-ocr

# Tag and push
aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
docker tag redactts-ocr:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/redactts-ocr:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/redactts-ocr:latest

# Create Lambda from container
aws lambda create-function \
  --function-name RedacTTS-OCR \
  --package-type Image \
  --code ImageUri=<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/redactts-ocr:latest \
  --role arn:aws:iam::<ACCOUNT_ID>:role/<LAMBDA_ROLE> \
  --memory-size 1024 \
  --timeout 300 \
  --environment Variables={S3_BUCKET=<BUCKET_NAME>}
```

## Testing
```bash
aws lambda invoke \
  --function-name RedacTTS-OCR \
  --payload '{"pdf_key": "uploads/test.pdf"}' \
  response.json
```

## Environment Variables
- `S3_BUCKET`: Your S3 bucket name
