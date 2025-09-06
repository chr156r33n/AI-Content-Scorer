# agents.md

## Purpose
This agent is responsible for building and maintaining a **Streamlit app** that takes a passage of text and highlights four features:  

1. **Semantic Triplets (Subject–Predicate–Object)**  
2. **Hedging Language**  
3. **Mixed Concepts / Topic Drift**  
4. **Passages That Are Too Long**  

The app must remain **lightweight** (no heavy transformer models, no external APIs), fast enough for interactive use, and easy to extend.

---

## Core Requirements

### Input
- User pastes or uploads a passage of text.  
- Optionally: multiple passages separated by blank lines.

### Output
- The text displayed with **inline highlights**:
  - **Subject** = green
  - **Predicate** = cyan
  - **Object** = purple
  - **Hedging** = amber
  - **Topic drift** = blue
  - **Too long** = red  

- A **summary panel** with:
  - Sentence/paragraph counts
  - Hedging count
  - Approx. number of S–P–O triples
  - Number of drift sentences
  - Number of "too long" spans
  - Readability metrics (if `textstat` available)

---

## Implementation Details

### Dependencies
- `streamlit`  
- `spacy` (small English model `en_core_web_sm`)  
- `scikit-learn` (for TF-IDF + cosine similarity)  
- `textstat` (optional readability)  

### Semantic Triplets
- Use spaCy dependency parsing.  
- Rule: sentence **root verb** with an **nsubj** (subject) and an **obj/dobj/attr/pobj** (object).  
- Extract spans for S, P, and O; highlight them.  
- Keep it **heuristic, not perfect** (speed > coverage).

### Hedging
- Maintain an editable **lexicon** of hedge terms (stored in sidebar).  
- Default list: "might, may, could, suggests, possibly, typically, generally, arguably, reportedly, allegedly, apparently…"  
- Highlight matches case-insensitively.

### Mixed Concepts / Drift
- Split text into sentences.  
- Represent each sentence with TF-IDF (unigrams + bigrams).  
- Compute cosine similarity between each sentence and the previous one.  
- If similarity < threshold (configurable slider), mark as drift.

### Overlong Passages
- Sentence-level: if word count > configurable threshold (default 35).  
- Paragraph-level: if word count > configurable threshold (default 180).  
- Highlight entire span as "too long".

---

## UI / UX
- Use **Streamlit sidebar** for settings:
  - Hedge lexicon editor
  - Sentence/paragraph thresholds
  - Topic drift threshold
- Use a **legend** above results showing color coding.  
- Allow user to **toggle analysis** with a button.  
- Show **readability metrics** (if `textstat` present) in a JSON summary block.  
- Provide tips at the bottom on tuning thresholds.

---

## Constraints
- Must run **locally** without internet calls.  
- Use only **lightweight NLP methods** (no GPT API calls, no BERT embeddings by default).  
- Keep highlights stable across overlapping tags.  
- Code must be clean, documented, and modular (easy to extend).  

---

## Extensions (Future Work)
- Export annotations as CSV/JSON.  
- Add toggle to enable SBERT embeddings instead of TF-IDF for drift detection.  
- Expand triplet extraction with custom spaCy `Matcher` patterns.  
- Integrate file upload for batch analysis.  