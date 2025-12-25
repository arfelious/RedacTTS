"""
Lambda Handler for PDF OCR Processing

Downloads PDF from S3, extracts text with redaction markers,
converts to Q&A JSON, and uploads results back to S3.

Environment Variables:
    S3_BUCKET: Target S3 bucket for file storage

Event: {"pdf_key": "uploads/document.pdf"}
Returns: {"qa_json_key": "...", "text_key": "...", "item_count": n}
"""

import json
import os
import tempfile
import uuid
import boto3

from pdf_redaction_extractor import PDFRedactionExtractor
from extract_qa import extract_qa

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """Process PDF and extract Q&A pairs."""
    bucket = os.environ.get('S3_BUCKET')
    if not bucket:
        return {'statusCode': 500, 'body': json.dumps({'error': 'S3_BUCKET not configured'})}
    
    pdf_key = event.get('pdf_key')
    if not pdf_key:
        return {'statusCode': 400, 'body': json.dumps({'error': 'pdf_key required'})}
    
    print(f"Processing: s3://{bucket}/{pdf_key}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_local = os.path.join(tmpdir, 'input.pdf')
        s3.download_file(bucket, pdf_key, pdf_local)
        
        extractor = PDFRedactionExtractor(dpi=300, debug=False)
        extracted_text = extractor.extract_text_with_redactions(pdf_local)
        
        text_local = os.path.join(tmpdir, 'extracted.txt')
        with open(text_local, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        qa_result = extract_qa(text_local)
        qa_json = json.dumps(qa_result, indent=2)
        
        # Use unique ID to prevent concurrent request conflicts
        unique_id = str(uuid.uuid4())[:8]
        base_name = pdf_key.replace('.pdf', '').replace('.PDF', '')
        
        text_key = f"{base_name}_{unique_id}_extracted.txt"
        s3.put_object(Bucket=bucket, Key=text_key, Body=extracted_text.encode('utf-8'))
        
        qa_key = f"{base_name}_{unique_id}_extracted_qa.json"
        s3.put_object(Bucket=bucket, Key=qa_key, Body=qa_json.encode('utf-8'))
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'qa_json_key': qa_key,
            'text_key': text_key,
            'item_count': len(qa_result)
        })
    }


if __name__ == "__main__":
    test_event = {"pdf_key": "uploads/test.pdf"}
    os.environ['S3_BUCKET'] = 'your-test-bucket'
    print(json.dumps(lambda_handler(test_event, None), indent=2))
