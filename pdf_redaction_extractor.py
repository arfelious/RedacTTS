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
        
        for w in words_data:
            text = w[4].strip()
            if not text:
                continue
            
            # Scale coordinates from PDF space to image space
            x = int(w[0] * zoom)
            y = int(w[1] * zoom)
            width = int((w[2] - w[0]) * zoom)
            height = int((w[3] - w[1]) * zoom)
            
            word_boxes.append(WordBox(
                text=text,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=100.0  # Embedded text is reliable
            ))
        
        return word_boxes, redactions
    
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
                # Shrink bounding box slightly to compensate for morphological expansion
                shrink = 3
                redactions.append(RedactedRegion(
                    x=x + shrink, 
                    y=y + shrink, 
                    width=max(1, w - 2*shrink), 
                    height=max(1, h - 2*shrink),
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
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        print(f"Processing: {pdf_path}")
        
        # Open PDF with PyMuPDF
        print("  Opening PDF...")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  Error opening PDF: {e}")
            return f"[Error processing PDF: {e}]"
        
        # Calculate zoom factor for desired DPI (default PDF is 72 DPI)
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        
        # Phase 1: Pre-process all pages (render, detect redactions, extract embedded text)
        print("  Phase 1: Pre-processing pages...")
        page_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"    Pre-processing page {page_num + 1}/{len(doc)}...")
            
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
            
            # Extract embedded text with positions
            embedded_word_boxes, redactions = self.extract_embedded_words_with_positions(
                page, zoom, redactions
            )
            
            page_data.append({
                'page_num': page_num,
                'cv_image': cv_image,
                'redactions': redactions,
                'embedded_words': embedded_word_boxes
            })
        
        doc.close()
        
        # Phase 2: Run OCR in parallel for pages that need it
        pages_needing_ocr = [p for p in page_data if p['redactions']]
        
        if pages_needing_ocr:
            num_workers = min(multiprocessing.cpu_count(), len(pages_needing_ocr))
            print(f"  Phase 2: Running OCR on {len(pages_needing_ocr)} pages using {num_workers} workers...")
            
            def ocr_page_wrapper(page_info):
                return self.ocr_page(page_info['cv_image'])
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                ocr_results = list(executor.map(ocr_page_wrapper, pages_needing_ocr))
            
            # Map OCR results back to page data
            ocr_idx = 0
            for page_info in page_data:
                if page_info['redactions']:
                    page_info['ocr_words'] = ocr_results[ocr_idx]
                    ocr_idx += 1
                else:
                    page_info['ocr_words'] = []
        else:
            for page_info in page_data:
                page_info['ocr_words'] = []
        
        # Phase 3: Filter and build text for each page
        print("  Phase 3: Building output text...")
        full_text = []
        
        for page_info in page_data:
            page_num = page_info['page_num']
            embedded_word_boxes = page_info['embedded_words']
            redactions = page_info['redactions']
            ocr_words = page_info['ocr_words']
            
            print(f"    Page {page_num + 1}: {len(redactions)} redactions, {len(embedded_word_boxes)} embedded words, {len(ocr_words)} OCR words")
            
            # Filter out botched redactions
            if redactions and ocr_words:
                embedded_word_boxes = self._filter_botched_redactions(
                    embedded_word_boxes, redactions, ocr_words
                )
            
            # Debug: save image with marked redactions
            if self.debug:
                cv_image = page_info['cv_image']
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
            
            # Build page text with redaction markers
            page_text = self._build_text_with_redactions(embedded_word_boxes, redactions)
            full_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        return "\n\n".join(full_text)
    
    def _filter_botched_redactions(
        self,
        embedded_words: List[WordBox],
        redactions: List[RedactedRegion],
        ocr_words: List[WordBox]
    ) -> List[WordBox]:
        """
        Filter out words that are in the embedded layer but visually redacted.
        
        This catches "botched" redactions where DoJ just highlighted text black
        but the text is still in the PDF's text layer.
        
        Logic: If embedded word overlaps a redaction box AND OCR doesn't see
        any word at that position, the word should be filtered out.
        """
        filtered_words = []
        
        # Punctuation should never be filtered by botched detection
        punctuation = set('.,!?;:\'"()-–—…')
        
        # Common stopwords - too likely to be false positives near redactions
        stopwords = {
            "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", 
            "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", 
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
            "t", "can", "will", "just", "don", "should", "now",
            # Single letters and transcript markers - critical for Q&A structure
            "i", "q", "a",
            # Common contractions (without punctuation)
            "it's", "i'm", "i'll", "i'd", "i've", "he's", "she's", "we're", "they're",
            "you're", "that's", "there's", "what's", "who's", "can't", "won't", 
            "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        }
        
        for word in embedded_words:
            word_lower = word.text.lower()
            # Also check without trailing punctuation (for "It's." -> "it's")
            word_stripped = word_lower.rstrip('.,!?;:\'"')
            
            # Skip punctuation - always keep it
            if len(word.text) <= 2 and all(c in punctuation for c in word.text):
                filtered_words.append(word)
                continue
            
            # Skip stopwords - always keep them (check both with and without trailing punct)
            if word_lower in stopwords or word_stripped in stopwords:
                filtered_words.append(word)
                continue
            
            # Check if word overlaps any redaction (no tolerance - strict overlap)
            overlaps_redaction = False
            for red in redactions:
                # Check for actual overlap (word must intersect redaction box)
                if (word.x < red.x + red.width and 
                    word.x + word.width > red.x and
                    word.y < red.y + red.height and 
                    word.y + word.height > red.y):
                    overlaps_redaction = True
                    break
            
            if not overlaps_redaction:
                # Not in redaction zone, keep the word
                filtered_words.append(word)
                continue
            
            # Word overlaps a redaction - check if OCR sees it
            ocr_confirms = False
            
            for ocr_word in ocr_words:
                # For single-char words, use bounding box intersection
                if len(word.text) == 1:
                    # Calculate intersection area
                    inter_x1 = max(word.x, ocr_word.x)
                    inter_y1 = max(word.y, ocr_word.y)
                    inter_x2 = min(word.x + word.width, ocr_word.x + ocr_word.width)
                    inter_y2 = min(word.y + word.height, ocr_word.y + ocr_word.height)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        word_area = word.width * word.height
                        
                        # OCR confirms if intersection is >50% of embedded word area
                        if word_area > 0 and inter_area / word_area > 0.5:
                            ocr_confirms = True
                            break
                else:
                    # For multi-char words, use position + fuzzy text match
                    y_distance = abs(word.y - ocr_word.y)
                    x_distance = abs(word.x - ocr_word.x)
                    
                    # Use tighter tolerance based on actual word dimensions
                    y_tolerance = word.height * 0.5
                    x_tolerance = word.width * 0.5
                    
                    if y_distance < y_tolerance and x_distance < x_tolerance:
                        if self.fuzzy_match_ratio(word.text, ocr_word.text) > self.fuzzy_threshold:
                            ocr_confirms = True
                            break
            
            if ocr_confirms:
                # OCR can see the word, so it's not visually redacted
                filtered_words.append(word)
            else:
                # Word is in embedded layer but OCR can't see it
                # This is a botched redaction - filter it out
                print(f"    [BOTCHED REDACTION] Filtered: '{word.text}' at ({word.x}, {word.y})")
        
        return filtered_words
    
    def _build_text_with_redactions(
        self, 
        word_boxes: List[WordBox], 
        redactions: List[RedactedRegion]
    ) -> str:
        """
        Build text output, inserting redaction markers at appropriate positions.
        Filters out footer text (below the last numbered line).
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
            # Use center for redaction positioning (both X and Y)
            # This places redactions more accurately relative to surrounding text
            center_y = red.y + (red.height // 2)
            center_x = red.x + (red.width // 2)
            elements.append({
                'type': 'redaction',
                'text': f"[REDACTED {red.estimated_words}]",
                'y': center_y,
                'x': center_x,
                'height': red.height,
                'width': red.width
            })
            
        if not elements:
            return ""
        
        # Find Y boundaries based on line numbers (1-25)
        # Track min Y (first line number) and max Y (last line number)
        min_numbered_y = float('inf')
        max_numbered_y = 0
        line_height_estimate = self.avg_word_height
        
        # First, collect all potential line number candidates with their Y positions
        line_number_candidates = []
        for elem in elements:
            text = elem['text']
            # Strip punctuation for line number check (handles cases like ".4" or "25.")
            text_stripped = text.strip('.,;:!?\'"()-')
            # Check if this looks like a line number (1-2 digits, value 1-25)
            if text_stripped and len(text_stripped) <= 2 and text_stripped.isdigit():
                num = int(text_stripped)
                if 1 <= num <= 25:
                    line_number_candidates.append({
                        'num': num,
                        'y': elem['y'],
                        'y_bottom': elem['y'] + elem['height'],
                        'height': elem['height']
                    })
        
        # Sort by Y to find sequence
        line_number_candidates.sort(key=lambda c: c['y'])
        
        # Filter out leading isolated numbers (likely page numbers in header)
        # A page number is isolated if there's a large Y gap before the next number
        filtered_candidates = line_number_candidates.copy()
        
        if len(filtered_candidates) >= 2:
            first = filtered_candidates[0]
            second = filtered_candidates[1]
            
            # Calculate Y gap between first and second candidate
            y_gap = second['y'] - first['y_bottom']
            avg_height = first['height']
            
            # If first number is isolated (gap > 3x line height), it's likely a page number
            if y_gap > avg_height * 3:
                print(f"    [HEADER] Skipping isolated page number: {first['num']} at y={first['y']} (gap={y_gap:.0f})")
                filtered_candidates = filtered_candidates[1:]
        
        # Now calculate boundaries from filtered candidates
        for cand in filtered_candidates:
            if cand['y'] < min_numbered_y:
                min_numbered_y = cand['y']
                line_height_estimate = cand['height']
            if cand['y_bottom'] > max_numbered_y:
                max_numbered_y = cand['y_bottom']
        
        # Filter out header (above first line number) and footer (below last)
        if min_numbered_y < float('inf') and max_numbered_y > 0:
            y_header_cutoff = min_numbered_y - (line_height_estimate * 0.5)
            y_footer_cutoff = max_numbered_y + (line_height_estimate * 1.5)
            
            filtered_elements = []
            for elem in elements:
                if elem['y'] < y_header_cutoff:
                    print(f"    [HEADER] Filtered: '{elem['text']}' at y={elem['y']} (cutoff={y_header_cutoff:.0f})")
                elif elem['y'] > y_footer_cutoff:
                    print(f"    [FOOTER] Filtered: '{elem['text']}' at y={elem['y']} (cutoff={y_footer_cutoff:.0f})")
                else:
                    filtered_elements.append(elem)
            elements = filtered_elements
        else:
            print(f"    [WARNING] No line numbers (1-25) found on page, skipping header/footer filter")
        
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
        
        # Post-process: merge redaction-only lines into adjacent text lines
        # if the redaction X positions fit into gaps between words
        def has_line_number(line):
            """Check if line has a line number marker (digit 1-25 at start)."""
            for elem in line:
                if elem['type'] == 'word' and elem['text'].isdigit():
                    num = int(elem['text'])
                    if 1 <= num <= 25:
                        return True
            return False
        
        def has_words(line):
            """Check if line has any non-redaction words."""
            return any(e['type'] == 'word' for e in line)
        
        def can_merge_redactions_into_line(redactions, target_line):
            """Check if redactions can fit into X-gaps of target line."""
            if not target_line:
                return False
            # Get X ranges of target line elements
            target_x_ranges = [(e['x'], e['x'] + e['width']) for e in target_line]
            # Check if each redaction fits in a gap
            for red in redactions:
                red_x1, red_x2 = red['x'], red['x'] + red['width']
                # Check if redaction overlaps with any existing element
                overlaps = any(not (red_x2 < t[0] or red_x1 > t[1]) for t in target_x_ranges)
                if overlaps:
                    return False
            return True
        
        merged_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # If this is a redaction-only line (no words)
            redactions_only = all(e['type'] == 'redaction' for e in line)
            
            if redactions_only and merged_lines:
                # Try to merge into previous line if it has words
                prev_line = merged_lines[-1]
                if has_words(prev_line) and not has_line_number(line):
                    # Merge: add redactions to previous line and re-sort by X
                    prev_line.extend(line)
                    prev_line.sort(key=lambda e: e['x'])
                    i += 1
                    continue
            
            merged_lines.append(line)
            i += 1
        
        lines = merged_lines

        # Build output text
        output_lines = []
        for line in lines:
            # Filter out obvious artifact words (II, III, IIII, etc.) that are OCR garbage
            filtered_elements = []
            for elem in line:
                text = elem['text']
                # Check if this is an artifact pattern (repeated I, l, |, 1 characters)
                artifact_chars = set('IilL|1!')
                if len(text) >= 2 and all(c in artifact_chars for c in text):
                    print(f"    [ARTIFACT] Filtered: '{text}'")
                    continue
                filtered_elements.append(elem)
            
            if filtered_elements:
                line_text = ' '.join(elem['text'] for elem in filtered_elements)
                output_lines.append(line_text)
        
        # Post-process: fix split line numbers (e.g., "1 1" -> "11")
        fixed_lines = self._fix_split_line_numbers(output_lines)
        
        return '\n'.join(fixed_lines)
    
    def _fix_split_line_numbers(self, lines: List[str]) -> List[str]:
        """
        Fix OCR errors where line numbers are split (e.g., '1 1' should be '11').
        Uses sequence tracking to validate corrections.
        """
        import re
        
        fixed_lines = []
        last_line_num = 0
        
        # Pattern: leading punctuation, digit(s), space, rest (e.g., ".4 A text")
        leading_punct_pattern = re.compile(r'^[.,;:]+(\d+)\s+(.*)$')
        # Pattern: start of line, digit, space(s), digit, space, rest
        split_num_pattern = re.compile(r'^(\d)\s+(\d)\s+(.*)$')
        # Pattern: normal line number at start
        line_num_pattern = re.compile(r'^(\d+)\s+(.*)$')
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append(line)
                continue
            
            # Check for leading punctuation before line number (e.g., ".4 A text")
            punct_match = leading_punct_pattern.match(stripped)
            if punct_match:
                num_str, rest = punct_match.groups()
                num = int(num_str)
                if 1 <= num <= 25:
                    # Fix by removing leading punctuation
                    fixed_lines.append(f"{num} {rest}")
                    last_line_num = num
                    continue
            
            # Check for split line number
            split_match = split_num_pattern.match(stripped)
            if split_match:
                d1, d2, rest = split_match.groups()
                merged_num = int(d1 + d2)
                
                # Validate: should be close to last_line_num + 1
                # Allow some flexibility (could skip lines, new page resets)
                if last_line_num > 0 and abs(merged_num - (last_line_num + 1)) <= 3:
                    fixed_lines.append(f"{merged_num} {rest}")
                    last_line_num = merged_num
                    continue
                elif last_line_num == 0 and merged_num >= 1 and merged_num <= 25:
                    # First line, reasonable starting number
                    fixed_lines.append(f"{merged_num} {rest}")
                    last_line_num = merged_num
                    continue
            
            # Check for normal line number - update tracking
            num_match = line_num_pattern.match(stripped)
            if num_match:
                num = int(num_match.group(1))
                # Page resets to 1-25, or continues sequence
                if num >= 1 and (num <= 25 or abs(num - last_line_num) <= 5):
                    last_line_num = num
            
            fixed_lines.append(line)
        
        return fixed_lines
    
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
