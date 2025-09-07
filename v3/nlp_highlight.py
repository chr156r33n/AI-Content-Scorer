"""
Pure-Python NLP highlighting module for semantic analysis.
Uses spaCy en_core_web_sm for lightweight processing.
"""

import spacy
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise ImportError("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")

# Hedging terms for detection
HEDGING_TERMS = {
    "might", "may", "could", "perhaps", "generally", "typically", "usually", 
    "seems", "appears", "suggests", "likely", "unlikely", "tends", 
    "sort of", "kind of", "around", "approximately", "roughly", 
    "often", "somewhat"
}

# Priority order for span de-overlapping (highest wins)
PRIORITY_ORDER = {
    "TooLong": 6,
    "Hedging": 5, 
    "Subject": 4,
    "Object": 3,
    "Predicate": 2,
    "TopicDrift": 1
}

def get_noun_chunks(doc: spacy.tokens.Doc) -> List[spacy.tokens.Span]:
    """Get all noun chunks from the document."""
    return list(doc.noun_chunks)

def find_subject_object_spans(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Find subject and object spans using noun chunks."""
    spans = []
    noun_chunks = get_noun_chunks(doc)
    
    for sent in doc.sents:
        # Find the root verb
        root = sent.root
        if root.pos_ != "VERB":
            continue
            
        # Find subject and object noun chunks
        subject_chunk = None
        object_chunk = None
        
        for chunk in noun_chunks:
            if chunk.sent == sent:  # Only consider chunks in this sentence
                # Check if any token in the chunk is a subject of the root
                for token in chunk:
                    if token.dep_ in ["nsubj", "nsubjpass"] and token.head == root:
                        subject_chunk = chunk
                        break
                
                # Check if any token in the chunk is an object of the root
                if not object_chunk:  # Only find first object
                    for token in chunk:
                        if token.dep_ in ["dobj", "pobj", "attr", "obj"] and token.head == root:
                            object_chunk = chunk
                            break
        
        # Add subject span
        if subject_chunk:
            spans.append({
                "label": "Subject",
                "start": subject_chunk.start_char,
                "end": subject_chunk.end_char,
                "text": subject_chunk.text
            })
        
        # Add object span
        if object_chunk:
            spans.append({
                "label": "Object", 
                "start": object_chunk.start_char,
                "end": object_chunk.end_char,
                "text": object_chunk.text
            })
    
    return spans

def find_predicate_spans(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Find predicate spans (root verb + modifiers, excluding subject/object)."""
    spans = []
    
    for sent in doc.sents:
        root = sent.root
        if root.pos_ != "VERB":
            continue
        
        # Start with root verb
        predicate_tokens = [root]
        
        # Find auxiliary verbs that are children of the root
        for token in sent:
            if (token.dep_ in ["aux", "auxpass"] and 
                token.head == root and 
                token.pos_ == "AUX" and
                token.i < root.i):  # Only auxiliaries before the root
                predicate_tokens.append(token)
        
        # Find adverbial modifiers of the verb
        for token in sent:
            if (token.dep_ in ["advmod", "neg"] and 
                token.head == root and
                token.pos_ in ["ADV", "PART"]):
                predicate_tokens.append(token)
        
        # Sort tokens by position
        predicate_tokens.sort(key=lambda t: t.i)
        
        # Create span from first to last token
        if predicate_tokens:
            start_idx = min(token.i for token in predicate_tokens)
            end_idx = max(token.i for token in predicate_tokens) + 1
            p_span = sent.doc[start_idx : end_idx]
            
            spans.append({
                "label": "Predicate",
                "start": p_span.start_char,
                "end": p_span.end_char,
                "text": p_span.text
            })
    
    return spans

def find_hedging_spans(text: str) -> List[Dict[str, Any]]:
    """Find hedging language spans."""
    spans = []
    text_lower = text.lower()
    
    for term in HEDGING_TERMS:
        term_lower = term.lower()
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(term_lower) + r'\b'
        
        for match in re.finditer(pattern, text_lower):
            # Find the original case version
            start = match.start()
            end = match.end()
            original_text = text[start:end]
            
            spans.append({
                "label": "Hedging",
                "start": start,
                "end": end,
                "text": original_text
            })
    
    return spans

def find_topic_drift_spans(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Find topic drift spans using TF-IDF similarity."""
    spans = []
    
    if len(list(doc.sents)) < 2:
        return spans
    
    # Get sentences and filter out empty ones
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    full_text = doc.text.strip()
    
    if len(sentences) < 2 or not full_text:
        return spans
    
    try:
        # Create TF-IDF vectors with error handling
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), 
            stop_words='english', 
            max_features=1000,
            min_df=1,  # Ensure we don't get empty matrices
            dtype=np.float64  # Explicitly set dtype
        )
        
        # Fit on all sentences + full text
        all_texts = sentences + [full_text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Check if we have a valid matrix
        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            return spans
        
        # L2 normalize
        norms = np.linalg.norm(tfidf_matrix.toarray(), axis=1, keepdims=True) + 1e-12
        tfidf_matrix = tfidf_matrix / norms
        
        # Get full text vector (last one)
        full_text_vector = tfidf_matrix[-1:]
        
        # Compare each sentence to full text
        for i, sent in enumerate(doc.sents):
            if i < len(sentences) and i < tfidf_matrix.shape[0] - 1:
                sentence_vector = tfidf_matrix[i:i+1]
                similarity = cosine_similarity(sentence_vector, full_text_vector)[0][0]
                
                if similarity < 0.15:
                    spans.append({
                        "label": "TopicDrift",
                        "start": sent.start_char,
                        "end": sent.end_char,
                        "text": sent.text
                    })
    
    except Exception as e:
        # If TF-IDF fails, skip topic drift detection
        print(f"Topic drift detection failed: {e}")
        return spans
    
    return spans

def find_too_long_spans(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Find sentences that are too long."""
    spans = []
    
    for sent in doc.sents:
        token_count = len(sent)
        char_count = len(sent.text)
        
        if token_count > 30 or char_count > 140:
            spans.append({
                "label": "TooLong",
                "start": sent.start_char,
                "end": sent.end_char,
                "text": sent.text
            })
    
    return spans

def deoverlap_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping spans, keeping highest priority ones."""
    if not spans:
        return spans
    
    # Sort by start position, then by priority (descending)
    spans.sort(key=lambda s: (s["start"], -PRIORITY_ORDER.get(s["label"], 0)))
    
    result = []
    for span in spans:
        # Check if this span overlaps with any already added span
        overlaps = False
        for existing in result:
            if (span["start"] < existing["end"] and 
                span["end"] > existing["start"]):
                overlaps = True
                break
        
        if not overlaps:
            result.append(span)
    
    return result

def annotate_passage(text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of spans:
    [{ "label": "Subject|Predicate|Object|Hedging|TopicDrift|TooLong",
       "start": int, "end": int, "text": str }]
    """
    if not text or not text.strip():
        return []
    
    try:
        # Process with spaCy
        doc = nlp(text)
        
        # Find all span types with error handling
        all_spans = []
        
        try:
            all_spans.extend(find_subject_object_spans(doc))
        except Exception as e:
            print(f"Subject/object detection failed: {e}")
        
        try:
            all_spans.extend(find_predicate_spans(doc))
        except Exception as e:
            print(f"Predicate detection failed: {e}")
        
        try:
            all_spans.extend(find_hedging_spans(text))
        except Exception as e:
            print(f"Hedging detection failed: {e}")
        
        try:
            all_spans.extend(find_topic_drift_spans(doc))
        except Exception as e:
            print(f"Topic drift detection failed: {e}")
        
        try:
            all_spans.extend(find_too_long_spans(doc))
        except Exception as e:
            print(f"Too long detection failed: {e}")
        
        # Remove overlaps
        final_spans = deoverlap_spans(all_spans)
        
        # Remove text field for final output (keep only label, start, end)
        return [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in final_spans]
    
    except Exception as e:
        print(f"Text analysis failed: {e}")
        return []

def render_html(text: str, spans: List[Dict[str, Any]]) -> str:
    """Render HTML with span wrappers and CSS classes."""
    if not spans:
        return text
    
    # Sort spans by start position
    spans = sorted(spans, key=lambda s: s["start"])
    
    result = []
    last_end = 0
    
    for span in spans:
        start = span["start"]
        end = span["end"]
        label = span["label"]
        
        # Add text before this span
        if start > last_end:
            result.append(text[last_end:start])
        
        # Add the span with CSS class
        span_text = text[start:end]
        css_class = f"hl-{label}"
        result.append(f'<span class="{css_class}" data-role="{label}">{span_text}</span>')
        
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        result.append(text[last_end:])
    
    return "".join(result)