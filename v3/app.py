import streamlit as st
import spacy
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import textstat
from typing import List, Dict, Tuple, Any
import json

# Page configuration
st.set_page_config(
    page_title="Text Analysis App",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching"""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
        return None

def extract_semantic_triplets(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Extract subject-predicate-object triplets from spaCy document"""
    triplets = []
    
    for sent in doc.sents:
        # Find the root verb
        root = sent.root
        if root.pos_ != "VERB":
            continue
            
        # Find subject and object
        subject = None
        obj = None
        
        for token in sent:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head == root:
                subject = token
            elif token.dep_ in ["dobj", "pobj", "attr", "obj"] and token.head == root:
                obj = token
        
        if subject and obj:
            triplets.append({
                'subject': {'text': subject.text, 'start': subject.idx, 'end': subject.idx + len(subject.text)},
                'predicate': {'text': root.text, 'start': root.idx, 'end': root.idx + len(root.text)},
                'object': {'text': obj.text, 'start': obj.idx, 'end': obj.idx + len(obj.text)},
                'sentence': sent.text
            })
    
    return triplets

def detect_hedging(text: str, hedge_terms: List[str]) -> List[Dict[str, Any]]:
    """Detect hedging language in text"""
    hedging_spans = []
    text_lower = text.lower()
    
    for term in hedge_terms:
        term_lower = term.lower()
        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            hedging_spans.append({
                'text': text[pos:pos + len(term)],
                'start': pos,
                'end': pos + len(term),
                'term': term
            })
            start = pos + 1
    
    return hedging_spans

def detect_topic_drift(sentences: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Detect topic drift between sentences using TF-IDF"""
    if len(sentences) < 2:
        return []
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate cosine similarity between consecutive sentences
    drift_sentences = []
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(tfidf_matrix[i-1:i], tfidf_matrix[i:i+1])[0][0]
        if similarity < threshold:
            drift_sentences.append({
                'sentence_index': i,
                'sentence': sentences[i],
                'similarity': similarity
            })
    
    return drift_sentences

def detect_overlong_passages(text: str, sentence_threshold: int = 35, paragraph_threshold: int = 180) -> List[Dict[str, Any]]:
    """Detect overlong sentences and paragraphs"""
    overlong_spans = []
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    current_pos = 0
    
    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            current_pos += len(paragraph) + 2
            continue
            
        # Check paragraph length
        word_count = len(paragraph.split())
        if word_count > paragraph_threshold:
            overlong_spans.append({
                'type': 'paragraph',
                'text': paragraph,
                'start': current_pos,
                'end': current_pos + len(paragraph),
                'word_count': word_count
            })
        
        # Check sentence lengths within paragraph
        sentences = re.split(r'[.!?]+', paragraph)
        sentence_start = current_pos
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            word_count = len(sentence.split())
            if word_count > sentence_threshold:
                overlong_spans.append({
                    'type': 'sentence',
                    'text': sentence.strip(),
                    'start': sentence_start,
                    'end': sentence_start + len(sentence),
                    'word_count': word_count
                })
            sentence_start += len(sentence) + 1
        
        current_pos += len(paragraph) + 2
    
    return overlong_spans

def highlight_text(text: str, highlights: Dict[str, List[Dict]]) -> str:
    """Create HTML with highlighted text"""
    # Sort all highlights by start position
    all_highlights = []
    for color, spans in highlights.items():
        for span in spans:
            all_highlights.append((span['start'], span['end'], color, span.get('text', '')))
    
    all_highlights.sort(key=lambda x: x[0])
    
    # Build highlighted HTML
    result = ""
    last_end = 0
    
    for start, end, color, span_text in all_highlights:
        # Add text before highlight
        if start > last_end:
            result += text[last_end:start]
        
        # Add highlighted text
        color_map = {
            'subject': 'background-color: #90EE90; color: #000;',  # Light green
            'predicate': 'background-color: #00FFFF; color: #000;',  # Cyan
            'object': 'background-color: #DDA0DD; color: #000;',  # Plum
            'hedging': 'background-color: #FFD700; color: #000;',  # Gold
            'drift': 'background-color: #87CEEB; color: #000;',  # Sky blue
            'overlong': 'background-color: #FFB6C1; color: #000;'  # Light pink
        }
        
        result += f'<span style="{color_map.get(color, "")}">{span_text}</span>'
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        result += text[last_end:]
    
    return result

def calculate_readability_metrics(text: str) -> Dict[str, float]:
    """Calculate readability metrics using textstat"""
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text)
        }
    except:
        return {}

def main():
    st.title("📝 Text Analysis App")
    st.markdown("Analyze text for semantic triplets, hedging language, topic drift, and overlong passages.")
    
    # Load spaCy model
    if st.session_state.nlp is None:
        with st.spinner("Loading spaCy model..."):
            st.session_state.nlp = load_spacy_model()
    
    if st.session_state.nlp is None:
        st.stop()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Hedging terms
    st.sidebar.subheader("Hedging Terms")
    default_hedge_terms = [
        "might", "may", "could", "suggests", "possibly", "typically", 
        "generally", "arguably", "reportedly", "allegedly", "apparently",
        "seems", "appears", "likely", "probably", "perhaps", "maybe",
        "tends to", "often", "usually", "sometimes", "frequently"
    ]
    
    hedge_terms_text = st.sidebar.text_area(
        "Edit hedging terms (one per line):",
        value="\n".join(default_hedge_terms),
        height=200
    )
    hedge_terms = [term.strip() for term in hedge_terms_text.split('\n') if term.strip()]
    
    # Thresholds
    st.sidebar.subheader("Thresholds")
    sentence_threshold = st.sidebar.slider("Sentence word threshold", 10, 100, 35)
    paragraph_threshold = st.sidebar.slider("Paragraph word threshold", 50, 500, 180)
    drift_threshold = st.sidebar.slider("Topic drift threshold", 0.1, 0.9, 0.3, 0.05)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        text_input = st.text_area(
            "Paste your text here:",
            height=300,
            placeholder="Enter text to analyze..."
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Process text with spaCy
                    doc = st.session_state.nlp(text_input)
                    
                    # Extract features
                    triplets = extract_semantic_triplets(doc)
                    hedging = detect_hedging(text_input, hedge_terms)
                    sentences = [sent.text for sent in doc.sents]
                    drift = detect_topic_drift(sentences, drift_threshold)
                    overlong = detect_overlong_passages(text_input, sentence_threshold, paragraph_threshold)
                    
                    # Store results
                    st.session_state.analysis_results = {
                        'triplets': triplets,
                        'hedging': hedging,
                        'drift': drift,
                        'overlong': overlong,
                        'text': text_input,
                        'sentences': sentences
                    }
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("📊 Summary")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Basic counts
            st.metric("Sentences", len(results['sentences']))
            st.metric("Paragraphs", len(text_input.split('\n\n')))
            st.metric("Hedging instances", len(results['hedging']))
            st.metric("S-P-O triplets", len(results['triplets']))
            st.metric("Topic drift", len(results['drift']))
            st.metric("Overlong spans", len(results['overlong']))
            
            # Readability metrics
            readability = calculate_readability_metrics(text_input)
            if readability:
                st.subheader("📈 Readability")
                for metric, value in readability.items():
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1f}")
    
    # Display results
    if st.session_state.analysis_results:
        st.subheader("🎨 Highlighted Text")
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 **Subject** - Green
        - 🔵 **Predicate** - Cyan  
        - 🟣 **Object** - Purple
        - 🟡 **Hedging** - Amber
        - 🔵 **Topic Drift** - Blue
        - 🔴 **Too Long** - Red
        """)
        
        results = st.session_state.analysis_results
        
        # Prepare highlights
        highlights = {
            'subject': [t['subject'] for t in results['triplets']],
            'predicate': [t['predicate'] for t in results['triplets']],
            'object': [t['object'] for t in results['triplets']],
            'hedging': results['hedging'],
            'overlong': results['overlong']
        }
        
        # Add drift highlights (approximate positions)
        drift_highlights = []
        for drift_item in results['drift']:
            sentence_idx = drift_item['sentence_index']
            if sentence_idx < len(results['sentences']):
                sentence = results['sentences'][sentence_idx]
                # Find approximate position in original text
                pos = results['text'].find(sentence)
                if pos != -1:
                    drift_highlights.append({
                        'start': pos,
                        'end': pos + len(sentence),
                        'text': sentence
                    })
        highlights['drift'] = drift_highlights
        
        # Create highlighted HTML
        highlighted_html = highlight_text(results['text'], highlights)
        
        # Display highlighted text
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            tab1, tab2, tab3, tab4 = st.tabs(["Triplets", "Hedging", "Drift", "Overlong"])
            
            with tab1:
                if results['triplets']:
                    for i, triplet in enumerate(results['triplets']):
                        st.write(f"**Triplet {i+1}:**")
                        st.write(f"- Subject: {triplet['subject']['text']}")
                        st.write(f"- Predicate: {triplet['predicate']['text']}")
                        st.write(f"- Object: {triplet['object']['text']}")
                        st.write(f"- Sentence: {triplet['sentence']}")
                        st.write("---")
                else:
                    st.write("No semantic triplets found.")
            
            with tab2:
                if results['hedging']:
                    for hedge in results['hedging']:
                        st.write(f"**{hedge['term']}** at position {hedge['start']}")
                else:
                    st.write("No hedging language detected.")
            
            with tab3:
                if results['drift']:
                    for drift in results['drift']:
                        st.write(f"**Sentence {drift['sentence_index']}:** (similarity: {drift['similarity']:.3f})")
                        st.write(drift['sentence'])
                        st.write("---")
                else:
                    st.write("No topic drift detected.")
            
            with tab4:
                if results['overlong']:
                    for overlong in results['overlong']:
                        st.write(f"**{overlong['type'].title()}** ({overlong['word_count']} words)")
                        st.write(overlong['text'])
                        st.write("---")
                else:
                    st.write("No overlong passages detected.")
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - **Adjust thresholds** in the sidebar to fine-tune detection sensitivity
    - **Edit hedging terms** to customize hedging language detection
    - **Lower drift threshold** to catch more subtle topic changes
    - **Increase word thresholds** to be more lenient with passage length
    """)

if __name__ == "__main__":
    main()