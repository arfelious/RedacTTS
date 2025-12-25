"""
PDF Text Extractor with Redaction Detection

This script extracts text from PDFs while detecting black redacted areas
and inserting [REDACTED <word_count>] markers with estimated word counts.

Requirements:
    pip install pymupdf opencv-python numpy pytesseract Pillow

"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from PIL import Image
from difflib import SequenceMatcher

# Storage abstraction for S3/local
try:
    import storage
except ImportError:
    storage = None  # Fallback to local-only if storage not available


@dataclass
class RedactedRegion:
    """Represents a detected redacted region in the PDF."""
    x: int
    y: int
    width: int
    height: int
    page_num: int
    estimated_words: int
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass 
class WordBox:
    """Represents a word detected by OCR with its bounding box."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PDFRedactionExtractor:
    """
    Extracts text from PDFs while detecting and marking redacted sections.
    
    Uses a combination of:
    - PyMuPDF for embedded text extraction
    - Tesseract OCR for word-level text recognition
    - OpenCV for detecting black redaction rectangles
    """
    
    # Average characters per word (including space)
    AVG_CHARS_PER_WORD = 5.5
    
    # Minimum aspect ratio for a redaction (width/height)
    # Helps filter out small squares that might be bullets or other elements
    MIN_REDACTION_ASPECT_RATIO = 1.5
    
    # Minimum area for a redaction (in pixels^2 at 300 DPI)
    MIN_REDACTION_AREA = 500
    
    # Maximum area for a redaction (to filter out full-page blacks)
    MAX_REDACTION_AREA_RATIO = 0.5  # Max 50% of page
    
    # Threshold for "black" detection (0-255, lower = stricter)
    BLACK_THRESHOLD = 30
    
    def __init__(self, dpi: int = 300, debug: bool = False):
        """
        Initialize the extractor.
        
        Args:
            dpi: Resolution for PDF to image conversion
            debug: If True, saves debug images showing detected redactions
        """
        self.dpi = dpi
        self.debug = debug
        # Estimated average word width in pixels at 300 DPI
        # Based on 12pt font (~16px height -> actually ~50px at 300DPI), avg 5 chars/word
        # Scale factor for DPI
        self.avg_word_width = int(40 * (dpi / 300))
        self.avg_word_height = int(50 * (dpi / 300))
        # Fuzzy matching threshold (0-1, higher = stricter)
        self.fuzzy_threshold = 0.6
        # Characters commonly produced as artifacts from redacted areas
        self.artifact_chars = set('IilL|1!/')
    
    def is_redaction_artifact(self, text: str) -> bool:
        """
        Check if a word looks like an artifact from a redacted area.
        
        These are typically repeated characters like "IIIIII", "llllll", "||||||"
        that appear when the PDF's text layer picks up garbage from black boxes.
        
        Args:
            text: The word text to check
            
        Returns:
            True if the word appears to be a redaction artifact
        """
        if len(text) < 2:
            return False
        
        # Check if word is entirely artifact characters (even short ones like "II", "III")
        all_artifact = all(c in self.artifact_chars for c in text)
        if all_artifact and len(text) >= 2:
            return True
        
        # Check if word is mostly the same character repeated
        unique_chars = set(text)
        
        # If 1-2 unique chars and mostly artifact characters, it's likely garbage
        if len(unique_chars) <= 2:
            artifact_count = sum(1 for c in text if c in self.artifact_chars)
            if artifact_count / len(text) >= 0.7:
                return True
        
        # Check for patterns like "IIIII", "lllll", "||||"
        if len(unique_chars) == 1 and text[0] in self.artifact_chars:
            return True
        
        return False
    
    def extract_embedded_words_with_positions(
        self, 
        page: fitz.Page, 
        zoom: float,
        redactions: List[RedactedRegion]
    ) -> Tuple[List[WordBox], List[RedactedRegion]]:
        """
        Extract embedded text words from a PDF page with positions scaled to image space.
        Filters out redaction artifacts and detects potential missed redactions.
        
        Args:
            page: PyMuPDF page object
            zoom: Zoom factor to scale coordinates to image space
            redactions: Already detected redactions (to check for false negatives)
            
        Returns:
            Tuple of (word_boxes, updated_redactions including any new ones found)
        """
        # Get text as words with positions
        words_data = page.get_text("words")
        
        # words_data format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # Sort by block, line, word to get reading order
        words_data = sorted(words_data, key=lambda w: (w[5], w[6], w[7]))
        
        word_boxes = []
        detected_artifact_regions = []
        
        for w in words_data:
            text = w[4].strip()
            if not text:
                continue
            
            # Scale coordinates from PDF space to image space
            x = int(w[0] * zoom)
            y = int(w[1] * zoom)
            width = int((w[2] - w[0]) * zoom)
            height = int((w[3] - w[1]) * zoom)
            
            # Check if this looks like a redaction artifact
            if self.is_redaction_artifact(text):
                # Check if there's already a detected redaction overlapping this position
                is_covered = False
                for red in redactions:
                    # Check for overlap
                    if (x < red.x + red.width and x + width > red.x and
                        y < red.y + red.height and y + height > red.y):
                        is_covered = True
                        break
                
                if not is_covered:
                    # This is a potential false negative - artifact without detected redaction
                    # Create a new redaction region for it
                    estimated_words = self._estimate_word_count(width, height)
                    detected_artifact_regions.append(RedactedRegion(
                        x=x, y=y, width=width, height=height,
                        page_num=0,  # Will be set by caller
                        estimated_words=max(1, estimated_words)
                    ))
                
                # Skip adding this artifact to word_boxes
                continue
            
            word_boxes.append(WordBox(
                text=text,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=100.0  # Embedded text is reliable
            ))
        
        # Merge artifact-detected redactions with existing ones
        all_redactions = list(redactions) + detected_artifact_regions
        if detected_artifact_regions:
            all_redactions = self._merge_redactions(all_redactions)
        
        return word_boxes, all_redactions
    
    def extract_embedded_words(self, page: fitz.Page, zoom: float) -> List[str]:
        """
        Extract embedded text words from a PDF page using PyMuPDF.
        
        Args:
            page: PyMuPDF page object
            zoom: Zoom factor to scale coordinates to image space
            
        Returns:
            List of words in reading order
        """
        # Get text as words with positions
        words_data = page.get_text("words")
        
        # words_data format: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # Sort by block, line, word to get reading order
        words_data = sorted(words_data, key=lambda w: (w[5], w[6], w[7]))
        
        # Extract just the text
        embedded_words = [w[4].strip() for w in words_data if w[4].strip()]
        
        return embedded_words
    
    def fuzzy_match_ratio(self, s1: str, s2: str) -> float:
        """
        Calculate fuzzy match ratio between two strings.
        """
        s1_norm = s1.lower().strip()
        s2_norm = s2.lower().strip()
        
        if not s1_norm or not s2_norm:
            return 0.0
        
        return SequenceMatcher(None, s1_norm, s2_norm).ratio()
    
    def detect_redactions(self, image: np.ndarray, page_num: int) -> List[RedactedRegion]:
        """
        Detect black redacted rectangles in an image.
        
        Args:
            image: OpenCV image (BGR format)
            page_num: Page number for reference
            
        Returns:
            List of detected RedactedRegion objects
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find very dark (black) regions
        _, binary = cv2.threshold(gray, self.BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up and connect nearby black regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        redactions = []
        page_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter based on size and aspect ratio
            if area < self.MIN_REDACTION_AREA:
                continue
            
            if area > page_area * self.MAX_REDACTION_AREA_RATIO:
                continue
            
            # Check if the region is actually filled (not just an outline)
            roi = binary[y:y+h, x:x+w]
            fill_ratio = np.sum(roi > 0) / (w * h)
            
            # Redactions should be mostly filled (>70%)
            if fill_ratio < 0.7:
                continue
            
            # Check aspect ratio - redactions are typically wider than tall
            # But we also want to catch multi-line redactions
            aspect_ratio = w / h if h > 0 else 0
            
            # Estimate word count based on dimensions
            estimated_words = self._estimate_word_count(w, h)
            
            if estimated_words >= 1:  # At least 1 word
                redactions.append(RedactedRegion(
                    x=x, y=y, width=w, height=h,
                    page_num=page_num,
                    estimated_words=estimated_words
                ))
        
        # Merge overlapping or adjacent redactions
        redactions = self._merge_redactions(redactions)
        
        return redactions
    
    def _estimate_word_count(self, width: int, height: int) -> int:
        """
        Estimate the number of words that would fit in a redacted region.
        
        For single-line redactions: based on width
        For multi-line redactions: based on area
        
        Args:
            width: Width of the redacted region in pixels
            height: Height of the redacted region in pixels
            
        Returns:
            Estimated word count
        """
        # Determine if this is likely a multi-line redaction
        # Multi-line if height is significantly more than a single line
        line_height = self.avg_word_height * 1.5  # Account for line spacing
        
        estimated_lines = max(1, height / line_height)
        
        if estimated_lines <= 1.3:
            # Single line: estimate based on width only
            words_per_line = width / self.avg_word_width
            return max(1, round(words_per_line))
        else:
            # Multi-line: estimate based on area
            # Account for partial last line
            full_lines = int(estimated_lines)
            partial_line = estimated_lines - full_lines
            
            words_per_line = width / self.avg_word_width
            total_words = (full_lines * words_per_line) + (partial_line * words_per_line * 0.5)
            
            return max(1, round(total_words))
    
    def _merge_redactions(self, redactions: List[RedactedRegion]) -> List[RedactedRegion]:
        """
        Merge overlapping or vertically adjacent redactions (multi-line redactions).
        
        Args:
            redactions: List of detected redactions
            
        Returns:
            Merged list of redactions
        """
        if not redactions:
            return redactions
        
        # Sort by y position then x position
        redactions = sorted(redactions, key=lambda r: (r.y, r.x))
        
        merged = []
        current = redactions[0]
        
        for next_red in redactions[1:]:
            # Check if redactions should be merged
            # They should be merged if:
            # 1. They overlap vertically (or are very close)
            # 2. They have similar x positions (same column of text)
            
            vertical_gap = next_red.y - (current.y + current.height)
            horizontal_overlap = (
                min(current.x + current.width, next_red.x + next_red.width) -
                max(current.x, next_red.x)
            )
            
            # Merge if vertically close and horizontally overlapping
            if vertical_gap < self.avg_word_height * 2 and horizontal_overlap > 0:
                # Expand current to include next
                new_x = min(current.x, next_red.x)
                new_y = min(current.y, next_red.y)
                new_x2 = max(current.x + current.width, next_red.x + next_red.width)
                new_y2 = max(current.y + current.height, next_red.y + next_red.height)
                
                new_width = new_x2 - new_x
                new_height = new_y2 - new_y
                
                current = RedactedRegion(
                    x=new_x, y=new_y,
                    width=new_width, height=new_height,
                    page_num=current.page_num,
                    estimated_words=self._estimate_word_count(new_width, new_height)
                )
            else:
                merged.append(current)
                current = next_red
        
        merged.append(current)
        return merged
    
    def ocr_page(self, image: np.ndarray) -> List[WordBox]:
        """
        Perform word-level OCR on an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            List of WordBox objects with text and positions
        """
        # Convert to RGB for Tesseract
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Get word-level OCR data
        ocr_data = pytesseract.image_to_data(
            pil_image, 
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )
        
        words = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # Skip empty or low-confidence words
            if not text or conf < 30:
                continue
            
            words.append(WordBox(
                text=text,
                x=ocr_data['left'][i],
                y=ocr_data['top'][i],
                width=ocr_data['width'][i],
                height=ocr_data['height'][i],
                confidence=conf
            ))
        
        return words
    
    def extract_text_with_redactions(self, pdf_path: str) -> str:
        """
        Extract text from a PDF, marking redacted sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text with [REDACTED <word_count>] markers
        """
        print(f"Processing: {pdf_path}")
        
        # Open PDF with PyMuPDF
        print("  Opening PDF...")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  Error opening PDF: {e}")
            return f"[Error processing PDF: {e}]"
        
        full_text = []
        
        # Calculate zoom factor for desired DPI (default PDF is 72 DPI)
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"  Processing page {page_num + 1}/{len(doc)}...")
            
            # Render page to image using PyMuPDF
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to numpy array (RGB format)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8)
            img_data = img_data.reshape(pix.height, pix.width, pix.n)
            
            # Convert RGB to BGR for OpenCV
            if pix.n == 4:  # RGBA
                cv_image = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                cv_image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            # Detect redactions from image (black boxes)
            redactions = self.detect_redactions(cv_image, page_num + 1)
            print(f"    Found {len(redactions)} redacted regions (from image)")
            
            # Extract embedded text with positions (higher quality than OCR)
            # Also filters out redaction artifacts and detects false negatives
            embedded_word_boxes, redactions = self.extract_embedded_words_with_positions(
                page, zoom, redactions
            )
            print(f"    Extracted {len(embedded_word_boxes)} embedded words from PDF")
            print(f"    Total redactions after artifact detection: {len(redactions)}")
            
            # Debug: save image with marked redactions
            if self.debug:
                debug_image = cv_image.copy()
                for red in redactions:
                    cv2.rectangle(debug_image, 
                                  (red.x, red.y), 
                                  (red.x + red.width, red.y + red.height),
                                  (0, 0, 255), 3)
                    cv2.putText(debug_image, f"~{red.estimated_words} words",
                               (red.x, red.y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                debug_path = f"debug_page_{page_num + 1}.png"
                cv2.imwrite(debug_path, debug_image)
                print(f"    Saved debug image: {debug_path}")
            
            # Build page text with redaction markers
            page_text = self._build_text_with_redactions(embedded_word_boxes, redactions)
            full_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        doc.close()
        return "\n\n".join(full_text)
    
    def _build_text_with_redactions(
        self, 
        word_boxes: List[WordBox], 
        redactions: List[RedactedRegion]
    ) -> str:
        """
        Build text output, inserting redaction markers at appropriate positions.
        realigning text based on Y-coordinates.
        """
        if not word_boxes and not redactions:
            return "[Empty page]"
        
        # Create a list of all elements
        elements = []
        
        for word_box in word_boxes:
            elements.append({
                'type': 'word',
                'text': word_box.text,
                'y': word_box.y,
                'x': word_box.x,
                'height': word_box.height,
                'width': word_box.width
            })
        
        for red in redactions:
            elements.append({
                'type': 'redaction',
                'text': f"[REDACTED {red.estimated_words}]",
                'y': red.y,
                'x': red.x,
                'height': red.height,
                'width': red.width
            })
            
        if not elements:
            return ""

        # Sort by Y primarily
        elements.sort(key=lambda e: e['y'])
        
        lines = []
        current_line = []
        
        # Robust Line Grouping
        # We use a dynamic tolerance based on the element height
        if elements:
            current_line = [elements[0]]
            current_line_y_sum = elements[0]['y']
            current_line_count = 1
            
            for elem in elements[1:]:
                # Calculate average Y of current line
                avg_y = current_line_y_sum / current_line_count
                
                # Check if this element belongs to the stored line
                # Tolerance: Half the height of the element or strict constant?
                # Using element height is safer for mixed font sizes.
                # If overlap is significant.
                
                # Simple logic: if vertical distance to average < half height
                height = elem['height']
                if abs(elem['y'] - avg_y) < (height * 0.5):
                    current_line.append(elem)
                    current_line_y_sum += elem['y']
                    current_line_count += 1
                else:
                    # New line
                    # Sort previous line by X
                    current_line.sort(key=lambda e: e['x'])
                    lines.append(current_line)
                    
                    # Start new
                    current_line = [elem]
                    current_line_y_sum = elem['y']
                    current_line_count = 1
            
            # Append last line
            if current_line:
                current_line.sort(key=lambda e: e['x'])
                lines.append(current_line)

        # Build output text
        output_lines = []
        for line in lines:
            # Check for large gaps (whitespace) to preserve columns? 
            # For now, just space join
            line_text = ' '.join(elem['text'] for elem in line)
            output_lines.append(line_text)
        
        return '\n'.join(output_lines)
    
    def process_path(self, path: str, output_dir: Optional[str] = None) -> None:
        """
        Process a PDF file or directory of PDF files.
        
        Args:
            path: Path to PDF file or directory
            output_dir: Optional output directory for extracted text files
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"Error: Path '{path}' does not exist.")
            return
            
        pdf_files = []
        if path_obj.is_file():
            if path_obj.suffix.lower() == '.pdf':
                pdf_files = [path_obj]
            else:
                print(f"Error: '{path}' is not a PDF file.")
                return
        elif path_obj.is_dir():
            pdf_files = list(path_obj.glob("*.pdf")) + list(path_obj.glob("*.PDF"))
        else:
            print(f"Error: '{path}' is not a file or directory.")
            return

        if not pdf_files:
            print(f"No PDF files found in '{path}'")
            return
            
        print(f"Found {len(pdf_files)} PDF file(s)")
        
        # Create output directory if specified
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = path_obj if path_obj.is_dir() else path_obj.parent
        
        # Process each PDF
        for pdf_file in pdf_files:
            try:
                extracted_text = self.extract_text_with_redactions(str(pdf_file))
                
                # Save to text file
                output_file = out_path / f"{pdf_file.stem}_extracted.txt"
                
                # Use storage module if available, otherwise local
                if storage:
                    storage.write_file(str(output_file), extracted_text)
                else:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                
                print(f"  Saved: {output_file}")
                print()
                
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs with redaction detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pdf_redaction_extractor.py /path/to/file.pdf
    python pdf_redaction_extractor.py /path/to/pdfs
    python pdf_redaction_extractor.py /path/to/pdfs --output /path/to/output
    python pdf_redaction_extractor.py /path/to/pdfs --dpi 150 --debug
        """
    )
    
    parser.add_argument(
        'path',
        help='PDF file or Directory containing PDF files to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for extracted text files (default: same as input)',
        default=None
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF rendering (higher = better quality but slower, default: 300)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug images showing detected redactions'
    )
    
    args = parser.parse_args()
    
    extractor = PDFRedactionExtractor(dpi=args.dpi, debug=args.debug)
    extractor.process_path(args.path, args.output)


if __name__ == "__main__":
    main()
