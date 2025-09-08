"""
Pure-Python NLP highlighting module for semantic analysis.
Uses spaCy en_core_web_sm for lightweight processing.
"""

import spacy
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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
    "TooLong": 7,
    "TooComplex": 6,
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
    """Find subject and object spans using noun chunks and individual tokens."""
    spans = []
    noun_chunks = get_noun_chunks(doc)
    
    for sent in doc.sents:
        # Find the root verb
        root = sent.root
        if root.pos_ != "VERB":
            continue
            
        # Find subject and object spans
        subject_span = None
        object_span = None
        
        # First try noun chunks
        for chunk in noun_chunks:
            if chunk.sent == sent:  # Only consider chunks in this sentence
                # Check if any token in the chunk is a subject of the root
                for token in chunk:
                    if token.dep_ in ["nsubj", "nsubjpass"] and token.head == root:
                        subject_span = {
                            "start": chunk.start_char,
                            "end": chunk.end_char,
                            "text": chunk.text
                        }
                        break
                
                # Check if any token in the chunk is an object of the root
                if not object_span:  # Only find first object
                    for token in chunk:
                        # Check for direct objects
                        if token.dep_ in ["dobj", "attr", "obj"] and token.head == root:
                            object_span = {
                                "start": chunk.start_char,
                                "end": chunk.end_char,
                                "text": chunk.text
                            }
                            break
                        # Check for prepositional objects (objects of prepositions that are children of the root)
                        elif token.dep_ == "pobj" and token.head.head == root:
                            object_span = {
                                "start": chunk.start_char,
                                "end": chunk.end_char,
                                "text": chunk.text
                            }
                            break
        
        # If no object found in noun chunks, try individual tokens
        if not object_span:
            for token in sent:
                # Check for direct objects
                if token.dep_ in ["dobj", "attr", "obj"] and token.head == root:
                    # Try to find the full noun phrase including determiners
                    start_token = token
                    end_token = token
                    
                    # Look backwards for determiners, adjectives
                    for i in range(token.i - 1, sent.start - 1, -1):
                        prev_token = doc[i]
                        if prev_token.pos_ in ["DET", "ADJ", "NUM"] and prev_token.head == token:
                            start_token = prev_token
                        else:
                            break
                    
                    # Look forwards for adjectives, nouns
                    for i in range(token.i + 1, sent.end):
                        next_token = doc[i]
                        if next_token.pos_ in ["ADJ", "NOUN", "PROPN"] and next_token.head == token:
                            end_token = next_token
                        else:
                            break
                    
                    object_span = {
                        "start": start_token.idx,
                        "end": end_token.idx + len(end_token.text),
                        "text": doc[start_token.idx:end_token.idx + len(end_token.text)].text
                    }
                    break
                # Check for prepositional objects
                elif token.dep_ == "pobj" and token.head.head == root:
                    # Similar logic for prepositional objects
                    start_token = token
                    end_token = token
                    
                    # Look backwards for determiners, adjectives
                    for i in range(token.i - 1, sent.start - 1, -1):
                        prev_token = doc[i]
                        if prev_token.pos_ in ["DET", "ADJ", "NUM"] and prev_token.head == token:
                            start_token = prev_token
                        else:
                            break
                    
                    # Look forwards for adjectives, nouns
                    for i in range(token.i + 1, sent.end):
                        next_token = doc[i]
                        if next_token.pos_ in ["ADJ", "NOUN", "PROPN"] and next_token.head == token:
                            end_token = next_token
                        else:
                            break
                    
                    object_span = {
                        "start": start_token.idx,
                        "end": end_token.idx + len(end_token.text),
                        "text": doc[start_token.idx:end_token.idx + len(end_token.text)].text
                    }
                    break
        
        # Add subject span
        if subject_span:
            spans.append({
                "label": "Subject",
                "start": subject_span["start"],
                "end": subject_span["end"],
                "text": subject_span["text"]
            })
        
        # Add object span
        if object_span:
            spans.append({
                "label": "Object", 
                "start": object_span["start"],
                "end": object_span["end"],
                "text": object_span["text"]
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
    """Find topic drift spans using simple word overlap similarity."""
    spans = []
    
    if len(list(doc.sents)) < 2:
        return spans
    
    # Get sentences and filter out empty ones
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    full_text = doc.text.strip()
    
    if len(sentences) < 2 or not full_text:
        return spans
    
    try:
        # Simple word-based similarity instead of TF-IDF
        def get_word_set(text):
            # Simple word tokenization
            words = set(text.lower().split())
            # Remove very short words
            return {w for w in words if len(w) > 2}
        
        full_text_words = get_word_set(full_text)
        
        # Compare each sentence to full text
        for sent in doc.sents:
            sentence_words = get_word_set(sent.text)
            
            if len(sentence_words) == 0:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(sentence_words & full_text_words)
            union = len(sentence_words | full_text_words)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity < 0.15:
                    spans.append({
                        "label": "TopicDrift",
                        "start": sent.start_char,
                        "end": sent.end_char,
                        "text": sent.text
                    })
    
    except Exception as e:
        # If topic drift detection fails, skip it
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

def find_too_complex_spans(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Find sentences that are too complex based on multiple metrics."""
    spans = []
    
    for sent in doc.sents:
        complexity_score = 0
        complexity_reasons = []
        
        # 1. Dependency depth (how many levels of nested phrases)
        max_depth = 0
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        
        if max_depth > 6:  # More than 6 levels of nesting
            complexity_score += 2
            complexity_reasons.append(f"deep nesting ({max_depth} levels)")
        
        # 2. Number of clauses (main + subordinate)
        clause_count = 0
        for token in sent:
            if token.dep_ in ["advcl", "acl", "relcl", "ccomp", "xcomp"]:
                clause_count += 1
        
        if clause_count > 2:  # More than 2 subordinate clauses
            complexity_score += 2
            complexity_reasons.append(f"many clauses ({clause_count})")
        
        # 3. Prepositional phrase density
        prep_count = sum(1 for token in sent if token.dep_ == "prep")
        prep_density = prep_count / len(sent) if len(sent) > 0 else 0
        
        if prep_density > 0.15:  # More than 15% prepositions
            complexity_score += 1
            complexity_reasons.append(f"many prepositions ({prep_count})")
        
        # 4. Conjunction density
        conj_count = sum(1 for token in sent if token.dep_ in ["cc", "conj"])
        conj_density = conj_count / len(sent) if len(sent) > 0 else 0
        
        if conj_density > 0.1:  # More than 10% conjunctions
            complexity_score += 1
            complexity_reasons.append(f"many conjunctions ({conj_count})")
        
        # 5. Average words per clause
        if clause_count > 0:
            words_per_clause = len(sent) / (clause_count + 1)  # +1 for main clause
            if words_per_clause > 12:  # More than 12 words per clause
                complexity_score += 1
                complexity_reasons.append(f"long clauses ({words_per_clause:.1f} words/clause)")
        
        # 6. Passive voice detection
        passive_count = sum(1 for token in sent if token.dep_ == "auxpass")
        if passive_count > 0:
            complexity_score += 1
            complexity_reasons.append("passive voice")
        
        # 7. Long sentence (already caught by TooLong, but add to complexity)
        if len(sent) > 25:  # More than 25 tokens
            complexity_score += 1
            complexity_reasons.append(f"long sentence ({len(sent)} tokens)")
        
        # Flag as too complex if score >= 2 (lowered threshold)
        if complexity_score >= 2:
            spans.append({
                "label": "TooComplex",
                "start": sent.start_char,
                "end": sent.end_char,
                "text": sent.text,
                "complexity_score": complexity_score,
                "reasons": complexity_reasons
            })
    
    return spans

def deoverlap_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping spans, but allow semantic spans to coexist with TooLong spans."""
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
                
                # Allow overlapping between predicates and hedging
                if ((span["label"] == "Predicate" and existing["label"] == "Hedging") or
                    (span["label"] == "Hedging" and existing["label"] == "Predicate")):
                    # Allow this overlap
                    continue
                
                # Allow semantic spans (Subject, Predicate, Object, Hedging) to coexist with TooLong/TooComplex
                semantic_spans = {"Subject", "Predicate", "Object", "Hedging"}
                warning_spans = {"TooLong", "TooComplex"}
                if ((span["label"] in semantic_spans and existing["label"] in warning_spans) or
                    (span["label"] in warning_spans and existing["label"] in semantic_spans)):
                    # Allow this overlap - semantic analysis can coexist with length/complexity warnings
                    continue
                
                # Normal overlap - use priority
                overlaps = True
                break
        
        if not overlaps:
            result.append(span)
    
    return result

def annotate_passage(text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of spans:
    [{ "label": "Subject|Predicate|Object|Hedging|TopicDrift|TooLong|TooComplex",
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
        
        try:
            all_spans.extend(find_too_complex_spans(doc))
        except Exception as e:
            print(f"Too complex detection failed: {e}")
        
        # Keep all spans (allow overlaps). Sort by start, then end for stability.
        all_spans_sorted = sorted(all_spans, key=lambda s: (s["start"], s["end"]))

        # Remove text field for final output (keep only label, start, end)
        return [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in all_spans_sorted]
    
    except Exception as e:
        print(f"Text analysis failed: {e}")
        return []

def render_html(text: str, spans: List[Dict[str, Any]]) -> str:
    """Render HTML with support for overlapping spans via segmentation.

    We split the text into minimal non-overlapping segments by all span boundaries.
    Each segment is wrapped once, with all covering labels applied as multiple CSS classes.
    """
    if not spans:
        return text

    # Normalize and sort spans
    normalized_spans = []
    for s in spans:
        start = max(0, min(len(text), s.get("start", 0)))
        end = max(start, min(len(text), s.get("end", start)))
        if start < end:
            normalized_spans.append({"start": start, "end": end, "label": s.get("label", "")})

    if not normalized_spans:
        return text

    # Collect all boundaries
    boundaries = set([0, len(text)])
    for s in normalized_spans:
        boundaries.add(s["start"]) 
        boundaries.add(s["end"])
    sorted_bounds = sorted(boundaries)

    # Build intervals between boundaries
    segments = []  # (seg_start, seg_end, [labels])
    for i in range(len(sorted_bounds) - 1):
        seg_start = sorted_bounds[i]
        seg_end = sorted_bounds[i + 1]
        if seg_start >= seg_end:
            continue
        covering_labels = []
        for s in normalized_spans:
            if s["start"] <= seg_start and s["end"] >= seg_end:
                covering_labels.append(s["label"])
        segments.append((seg_start, seg_end, covering_labels))

    # Render by segments
    result_parts: List[str] = []
    for seg_start, seg_end, labels in segments:
        seg_text = text[seg_start:seg_end]
        if not labels:
            result_parts.append(seg_text)
            continue
        # Deduplicate labels while preserving order
        seen = set()
        unique_labels = []
        for lbl in labels:
            if lbl not in seen:
                seen.add(lbl)
                unique_labels.append(lbl)

        classes = " ".join([f"hl-{lbl}" for lbl in unique_labels])
        roles_attr = ",".join(unique_labels)
        result_parts.append(f'<span class="{classes}" data-roles="{roles_attr}">{seg_text}</span>')

    return "".join(result_parts)