FROM public.ecr.aws/lambda/python:3.11

# Install build dependencies
RUN yum install -y \
    gcc gcc-c++ make \
    libpng-devel libjpeg-devel libtiff-devel zlib-devel \
    autoconf automake libtool pkgconfig \
    wget tar gzip \
    && yum clean all

# Disable pymupdf source build
ENV PYMUPDF_SETUP_MUPDF_TESSERACT=0
ENV PYMUPDF_SETUP_MUPDF_BUILD=0

# Install Python packages with pinned versions that have pre-built wheels
RUN pip install --no-cache-dir \
    "PyMuPDF==1.23.8" \
    "numpy<2.0" \
    opencv-python-headless \
    pytesseract \
    Pillow \
    boto3

# Build Leptonica
WORKDIR /tmp
RUN wget -q https://github.com/DanBloomberg/leptonica/releases/download/1.84.1/leptonica-1.84.1.tar.gz \
    && tar -xzf leptonica-1.84.1.tar.gz \
    && cd leptonica-1.84.1 \
    && ./configure --prefix=/usr/local --quiet \
    && make -j$(nproc) -s \
    && make install -s \
    && cd .. && rm -rf leptonica-1.84.1*

# Build Tesseract
RUN wget -q https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.3.3.tar.gz \
    && tar -xzf 5.3.3.tar.gz \
    && cd tesseract-5.3.3 \
    && ./autogen.sh \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig ./configure --prefix=/usr/local --quiet \
    && make -j$(nproc) -s \
    && make install -s \
    && cd .. && rm -rf tesseract-5.3.3 5.3.3.tar.gz

# Download English trained data
RUN mkdir -p /usr/local/share/tessdata \
    && wget -q -O /usr/local/share/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata

# Set environment
ENV TESSDATA_PREFIX=/usr/local/share/tessdata
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR ${LAMBDA_TASK_ROOT}

COPY pdf_redaction_extractor.py .
COPY extract_qa.py .
COPY lambda_handler.py .

CMD ["lambda_handler.lambda_handler"]