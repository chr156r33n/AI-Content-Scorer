import streamlit as st
import pandas as pd
import numpy as np
import textstat
from typing import List, Dict, Tuple, Any
import json
from nlp_highlight import annotate_passage, render_html

# Page configuration
st.set_page_config(
    page_title="Text Analysis App",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text using the new NLP highlighting module."""
    try:
        # Get spans from the NLP module
        spans = annotate_passage(text)
        
        # Count different types of spans
        span_counts = {}
        for span in spans:
            label = span["label"]
            span_counts[label] = span_counts.get(label, 0) + 1
        
        # Generate HTML with highlighting
        highlighted_html = render_html(text, spans)
        
        return {
            'spans': spans,
            'span_counts': span_counts,
            'highlighted_html': highlighted_html,
            'text': text
        }
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        return None

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
    
    # Add CSS styles
    st.markdown("""
    <style>
    /* Semantic labels now use stacked underlines computed inline via box-shadow.
       Keep classes for legend mapping; avoid background fills to prevent clutter. */
    .hl-Subject, .hl-Predicate, .hl-Object, .hl-Hedging, .hl-TopicDrift {
        background: transparent;
    }

    /* Warnings use subtle background / left border */
    .hl-TooLong {
        background: rgba(239, 68, 68, 0.12);
        border-radius: 2px;
    }
    .hl-TooComplex {
        border-left: 3px solid rgba(236, 72, 153, 0.6);
        padding-left: 2px;
    }
    /* Improve readability with slight spacing between words when many effects stack */
    .hl-Subject, .hl-Predicate, .hl-Object, .hl-Hedging, .hl-TopicDrift, .hl-TooLong, .hl-TooComplex {
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
                    # Analyze text using new module
                    results = analyze_text(text_input)
                    if results:
                        st.session_state.analysis_results = results
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.subheader("📊 Summary")
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Basic counts
            span_counts = results['span_counts']
            st.metric("Subjects", span_counts.get('Subject', 0))
            st.metric("Predicates", span_counts.get('Predicate', 0))
            st.metric("Objects", span_counts.get('Object', 0))
            st.metric("Hedging instances", span_counts.get('Hedging', 0))
            st.metric("Topic drift", span_counts.get('TopicDrift', 0))
            st.metric("Too long", span_counts.get('TooLong', 0))
            st.metric("Too complex", span_counts.get('TooComplex', 0))
            
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
        - 🟠 **Too Complex** - Orange
        """)
        
        results = st.session_state.analysis_results
        
        # Display highlighted text
        st.markdown(results['highlighted_html'], unsafe_allow_html=True)
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            spans = results['spans']
            
            # Group spans by type
            spans_by_type = {}
            for span in spans:
                label = span['label']
                if label not in spans_by_type:
                    spans_by_type[label] = []
                spans_by_type[label].append(span)
            
            # Display each type
            for label, type_spans in spans_by_type.items():
                st.subheader(f"{label} Spans ({len(type_spans)})")
                for i, span in enumerate(type_spans):
                    start = span['start']
                    end = span['end']
                    text_snippet = results['text'][start:end]
                    st.write(f"**{i+1}.** Position {start}-{end}: \"{text_snippet}\"")
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - **Subjects** are highlighted in green - these are the main noun phrases
    - **Predicates** are highlighted in cyan - these are verb phrases with auxiliaries
    - **Objects** are highlighted in purple - these are the target noun phrases
    - **Hedging** terms are highlighted in amber - words that express uncertainty
    - **Topic drift** sentences are highlighted in blue - sentences that don't fit the main topic
    - **Too long** sentences are highlighted in red - sentences over 30 tokens or 140 characters
    """)

if __name__ == "__main__":
    main()