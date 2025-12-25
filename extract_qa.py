"""
Q&A Extraction from Transcript Text

Extracts Question/Answer pairs and Extra speaker segments from 
formatted transcript text files with [REDACTED n] markers.
"""

import re
import json
import sys
import os

try:
    import storage
except ImportError:
    storage = None


def extract_qa(file_path):
    """Extract Q&A pairs from a transcript text file."""
    if storage:
        content = storage.read_text(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    # Split by page markers and filter to relevant pages
    raw_pages = re.split(r'(--- Page \d+ ---)', content)
    trigger_pattern = re.compile(r'\d+\s+[AQ]\s+\w+')
    
    valid_pages = []
    start_extraction = False
    
    for segment in raw_pages:
        if not segment.strip():
            continue
        if trigger_pattern.search(segment):
            start_extraction = True
        if start_extraction:
            valid_pages.append(segment)
            
    if not valid_pages:
        print("No matching pages found.")
        return []

    full_text = "\n".join(valid_pages)
    lines = full_text.splitlines()

    # Patterns
    qa_start_pattern = re.compile(r'^\s*(?:(?:\[REDACTED[^\]]*\]|\d+)\s+)?([AQ])\s+(.*)', re.MULTILINE)
    extra_pattern = re.compile(r'^\s*(\d+\s+)?([A-Z][A-Z\s]+):(.*)', re.MULTILINE)
    
    def get_next_label(start_idx, all_lines):
        """Look ahead to determine the next explicit label (Q, A, or Extra)."""
        for k in range(start_idx, len(all_lines)):
            s_line = all_lines[k].strip()
            if not s_line or s_line.startswith('--- Page'):
                continue
            m_start = qa_start_pattern.match(all_lines[k])
            if m_start:
                return m_start.group(1)
            m_extra = extra_pattern.match(s_line)
            if m_extra and m_extra.group(2).strip() not in ['A', 'Q']:
                return 'Extra'
        return None

    def merge_redactions(text):
        """Merge consecutive [REDACTED n] tags into a single summed tag."""
        def replacement(match):
            nums = map(int, re.findall(r'\[REDACTED (\d+)\]', match.group(0)))
            return f"[REDACTED {sum(nums)}] "
        pattern = re.compile(r'(?:\[REDACTED \d+\][\sI]*){2,}')
        return pattern.sub(replacement, text).strip()

    qa_list = []
    current_obj = None
    found_first_explicit = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        if not line_stripped or line_stripped.startswith('--- Page'):
            i += 1
            continue
            
        match_start = qa_start_pattern.match(line)
        
        # Wait for first explicit Q/A
        if not found_first_explicit:
            if match_start:
                found_first_explicit = True
            else:
                i += 1
                continue

        match_extra = extra_pattern.match(line_stripped)
        
        is_extra = False
        if match_extra and not match_start:
            name = match_extra.group(2).strip()
            if name not in ['A', 'Q']:
                is_extra = True
        
        if match_start:
            q_or_a = match_start.group(1)
            text = merge_redactions(match_start.group(2).strip())
            
            # Check for mislabeled Extra (e.g., "A GRAND JUROR: ...")
            speaker_match = re.match(r'^([A-Z][A-Z\s]+):(.*)', text)
            if q_or_a == 'A' and speaker_match:
                name = speaker_match.group(1).strip()
                content = speaker_match.group(2).strip()
                if current_obj:
                    qa_list.append(current_obj)
                current_obj = {'Extra': {'text': content, 'name': name}}
            elif q_or_a == 'Q':
                if current_obj:
                    qa_list.append(current_obj)
                current_obj = {'Question': {'text': text}}
            elif q_or_a == 'A':
                if current_obj and 'Question' in current_obj and 'Answer' not in current_obj:
                    current_obj['Answer'] = {'text': text}
                else:
                    if current_obj:
                        qa_list.append(current_obj)
                    current_obj = {'Answer': {'text': text}}
                    
        elif is_extra:
            name = match_extra.group(2).strip()
            content = merge_redactions(match_extra.group(3).strip())
            if current_obj:
                qa_list.append(current_obj)
            current_obj = {'Extra': {'text': content, 'name': name}}
                    
        else:
            # Continuation or implicit line
            continuation_match = re.match(r'^\s*(?:(?:\[REDACTED[^\]]*\]|\d+)\s+)?(.*)', line)
            clean_text = merge_redactions(continuation_match.group(1).strip() if continuation_match else line_stripped)
            
            next_label = get_next_label(i + 1, lines)
            target_type = None
            
            if current_obj and 'Extra' in current_obj:
                target_type = None
            elif next_label == 'A':
                target_type = 'Question'
            elif next_label == 'Q':
                target_type = 'Answer'
            
            if target_type == 'Question':
                if not current_obj or 'Answer' in current_obj or 'Extra' in current_obj:
                    if current_obj:
                        qa_list.append(current_obj)
                    current_obj = {'Question': {'text': clean_text}}
                else:
                    current_obj['Question']['text'] = merge_redactions(current_obj['Question']['text'] + " " + clean_text)
                    
            elif target_type == 'Answer':
                if current_obj and 'Question' in current_obj and 'Answer' not in current_obj:
                    current_obj['Answer'] = {'text': clean_text}
                elif current_obj and 'Answer' in current_obj:
                    current_obj['Answer']['text'] = merge_redactions(current_obj['Answer']['text'] + " " + clean_text)
                else:
                    if current_obj:
                        qa_list.append(current_obj)
                    current_obj = {'Answer': {'text': clean_text}}
            
            else:
                if current_obj:
                    if 'Extra' in current_obj:
                        current_obj['Extra']['text'] = merge_redactions(current_obj['Extra']['text'] + " " + clean_text)
                    elif 'Answer' in current_obj:
                        current_obj['Answer']['text'] = merge_redactions(current_obj['Answer']['text'] + " " + clean_text)
                    elif 'Question' in current_obj:
                        current_obj['Question']['text'] = merge_redactions(current_obj['Question']['text'] + " " + clean_text)
        
        i += 1

    if current_obj:
        qa_list.append(current_obj)
        
    return qa_list


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_qa.py <file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    result = extract_qa(file_path)
    
    output_path = file_path.replace('.txt', '_qa.json')
    
    if storage:
        storage.write_file(output_path, json.dumps(result, indent=4))
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        
    orphan_q = sum(1 for item in result if 'Question' in item and 'Answer' not in item)
    orphan_a = sum(1 for item in result if 'Answer' in item and 'Question' not in item)
            
    print(f"Extraction complete. Found {len(result)} items.")
    print(f"Orphan Questions: {orphan_q}, Orphan Answers: {orphan_a}")
    print(f"Saved to {output_path}")
