"""
Storage Abstraction Layer

Provides transparent file operations for local filesystem and S3.
Paths starting with "s3://" route to S3; otherwise local filesystem.

Environment Variables:
    STORAGE_BACKEND: "local" (default) or "s3"
    S3_BUCKET: Required when using S3 backend
"""

import os
from typing import Union

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client('s3')
    return _s3_client


def _get_backend():
    return os.environ.get('STORAGE_BACKEND', 'local').lower()


def _get_bucket():
    return os.environ.get('S3_BUCKET', '')


def is_s3_path(path: str) -> bool:
    """Check if path is an S3 URI."""
    return path.startswith('s3://')


def _parse_s3_path(path: str) -> tuple:
    """Parse s3://bucket/key into (bucket, key)."""
    if not path.startswith('s3://'):
        raise ValueError(f"Not an S3 path: {path}")
    parts = path[5:].split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def _to_storage_path(path: str) -> str:
    """Convert path based on backend config."""
    if is_s3_path(path) or os.path.isabs(path):
        return path
    
    if _get_backend() == 's3':
        bucket = _get_bucket()
        if not bucket:
            raise ValueError("S3_BUCKET not set")
        return f"s3://{bucket}/{path}"
    
    return path


def read_file(path: str) -> bytes:
    """Read file from storage."""
    path = _to_storage_path(path)
    
    if is_s3_path(path):
        bucket, key = _parse_s3_path(path)
        return _get_s3_client().get_object(Bucket=bucket, Key=key)['Body'].read()
    
    with open(path, 'rb') as f:
        return f.read()


def read_text(path: str, encoding: str = 'utf-8') -> str:
    """Read text file from storage."""
    return read_file(path).decode(encoding)


def write_file(path: str, data: Union[bytes, str], encoding: str = 'utf-8'):
    """Write file to storage."""
    path = _to_storage_path(path)
    
    if isinstance(data, str):
        data = data.encode(encoding)
    
    if is_s3_path(path):
        bucket, key = _parse_s3_path(path)
        _get_s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)


def file_exists(path: str) -> bool:
    """Check if file exists."""
    path = _to_storage_path(path)
    
    if is_s3_path(path):
        bucket, key = _parse_s3_path(path)
        try:
            _get_s3_client().head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    
    return os.path.exists(path)


def get_local_path(path: str) -> str:
    """Get local filesystem path, downloading from S3 if needed."""
    import tempfile
    
    path = _to_storage_path(path)
    
    if is_s3_path(path):
        data = read_file(path)
        _, ext = os.path.splitext(path)
        fd, local_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(local_path, 'wb') as f:
            f.write(data)
        return local_path
    
    return path
