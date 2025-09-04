import math, gzip, io, re
from typing import List, Tuple
import numpy as np
import streamlit as st

# ---- Fast, torch-free embeddings ----
from fastembed import TextEmbedding

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    # First call downloads a small ONNX (~50–90MB) into HF cache; later runs are instant
    return TextEmbedding(model_name=model_name)

STOP = {"the","a","an","and","or","but","if","then","so","as","at","by","for","in","of","on","to","with",
        "is","are","was","were","be","been","being","it","its","this","that","these","those","we","you",
        "your","our","their","from","over","into","out","up","down","about","than","too","very"}
TOKEN_SPLIT = re.compile(r"[^\w'-]+", re.UNICODE)
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")

def tokenize(t: str) -> List[str]:
    return [x for x in TOKEN_SPLIT.split(t.lower()) if x]

def content_tokens(ts: List[str]) -> List[str]:
    return [t for t in ts if t not in STOP]

def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(raw)
    return len(buf.getvalue()) / max(1, len(raw))

def semantic_uniques_score(text: str) -> Tuple[float,int,int,int]:
    toks = tokenize(text)
    ctoks = content_tokens(toks)
    uniques = {t for t in ctoks if len(t) >= 3}
    score = (len(uniques)/max(1,len(ctoks))) if ctoks else 0.0
    return score, len(toks), len(ctoks), len(uniques)

def squash_01(x: float) -> float:
    return 1.0 - math.exp(-3.0 * max(0.0, x))

def geometric_mean(vals: List[float]) -> float:
    vals = [max(1e-9, min(1.0, v)) for v in vals]
    return float(np.exp(np.mean(np.log(vals))))

def split_sents(text: str):
    sents, idx = [], 0
    parts = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
    if not parts: return [(0, len(text), text)]
    for p in parts:
        start = text.find(p, idx); end = start + len(p)
        sents.append((start, end, p)); idx = end
    return sents

def sliding_windows(sents, win_size=3, stride=2):
    wins = []
    for i in range(0, len(sents), stride):
        chunk = sents[i:i+win_size]
        if not chunk: continue
        start, end = chunk[0][0], chunk[-1][1]
        wins.append((start, end, " ".join([c[2] for c in chunk])))
    return wins

def embed_fastembed(texts: List[str], model_name="BAAI/bge-small-en-v1.5") -> np.ndarray:
    model = get_embedder(model_name)
    # FastEmbed returns a generator of vectors
    vecs = np.stack(list(model.embed(texts))).astype(np.float32)
    # L2-normalize for cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # already normalized

def overlap_embed(passage: str, queries: List[str], model_name="BAAI/bge-small-en-v1.5",
                  win_size=3, stride=2):
    sents = split_sents(passage)
    wins = sliding_windows(sents, win_size=win_size, stride=stride) or [(0, len(passage), passage)]
    w_texts = [w[2] for w in wins]; w_spans = [(w[0], w[1]) for w in wins]

    vecs = embed_fastembed(queries + w_texts, model_name=model_name)
    q_vecs, w_vecs = vecs[:len(queries)], vecs[len(queries):]
    sims = cosine_sim(q_vecs, w_vecs) if len(queries) and len(w_texts) else np.zeros((0,0))

    per_q_max = np.max(sims, axis=1) if sims.size else np.array([0.0]*len(queries))
    overlap_raw = float(np.mean(per_q_max)) if len(per_q_max) else 0.0

    tokens = len(tokenize(passage))
    length_factor = min(1.0, math.log1p(tokens)/math.log1p(40.0))
    overlap_len = max(0.0, min(1.0, overlap_raw * length_factor))

    per_win_max = np.max(sims, axis=0) if sims.size else np.zeros(len(wins))
    window_scores = [(span[0], span[1], float(v)) for span, v in zip(w_spans, per_win_max)]
    q_labels = [f"Q{i+1}" for i in range(len(queries))]
    w_labels = [f"W{i+1}" for i in range(len(wins))]
    return overlap_len, window_scores, sims, q_labels, w_labels

def color_for_score(v: float) -> str:
    v = max(0.0, min(1.0, v))
    # Convert score to border width (0-3px) and opacity
    width = max(1, int(v * 3))  # 1-3px border width
    opacity = max(0.3, v)       # 30%-100% opacity
    return f"border: {width}px dotted rgba(255, 0, 0, {opacity:.2f}); padding: 2px; margin: 1px;"

def get_unique_words(text: str) -> set:
    """Get set of unique content words (non-stop, ≥3 chars, appearing only once)"""
    toks = tokenize(text)
    ctoks = content_tokens(toks)
    # Count occurrences of each content token
    from collections import Counter
    counts = Counter(ctoks)
    # Return only words that appear exactly once and are ≥3 chars
    return {word for word, count in counts.items() if count == 1 and len(word) >= 3}

def render_highlighted(passage: str, window_scores):
    if not passage: return ""
    
    # Get unique words for blue highlighting
    unique_words = get_unique_words(passage)
    
    scores = np.zeros(len(passage), float)
    for s, e, v in window_scores:
        s, e = max(0,s), min(len(passage), e)
        if e > s: scores[s:e] = np.maximum(scores[s:e], v)
    
    html, i = [], 0
    buck = lambda x: round(x, 2)
    while i < len(passage):
        b = buck(scores[i]); j = i+1
        while j < len(passage) and buck(scores[j]) == b: j += 1
        seg = (passage[i:j].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        
        # Check if this segment contains unique words
        seg_lower = seg.lower()
        # Extract actual words from segment for precise matching
        import re
        seg_words = re.findall(r"\b\w+\b", seg_lower)
        # Only highlight if the segment is exactly a single unique word
        # Count unique words in segment
        unique_count = sum(1 for word in seg_words if word in unique_words)
        is_unique = unique_count == 1 and len(seg_words) == 1
        
        if b > 0.1:  # High overlap score
            annotation = f"<sup style='font-size:0.7em; color:#666;'>({b:.2f})</sup>"
            if is_unique:
                # Both high overlap AND unique - combine styling
                html.append(f"<span style='{color_for_score(b)}; color: blue;'>{seg}{annotation}</span>")
            else:
                # Just high overlap
                html.append(f"<span style='{color_for_score(b)}'>{seg}{annotation}</span>")
        elif is_unique:
            # Just unique (no high overlap)
            html.append(f"<span style='color: blue; font-weight: bold;'>{seg}</span>")
        else:
            # Neither high overlap nor unique
            html.append(seg)
        i = j
    return "".join(html)

# ---- UI ----
st.set_page_config(page_title="Semantic Overlap & Density (FastEmbed)", layout="wide")
st.title("Semantic Overlap & Density — FastEmbed (no Torch)")
st.caption("CPU-only ONNX embeddings. Quick startup, solid quality. Great for local/air-gapped use.")

with st.sidebar:
    model_name = st.selectbox(
        "Embedding model",
        [
            "BAAI/bge-small-en-v1.5",   # 384-dim, fast & accurate
            "intfloat/e5-small-v2"      # another solid small model
        ],
        index=0
    )
    win_size = st.slider("Sentence window size", 1, 6, 3)
    stride   = st.slider("Window stride", 1, 6, 2)

colA, colB = st.columns([1,1])
with colA:
    passage = st.text_area("Passage", height=220, placeholder="Paste your content here…")
with colB:
    raw_queries = st.text_area("Queries (one per line, up to 10)", height=220,
                               placeholder="e.g.\nluxury resort whistler\nski-in ski-out suites\nspa and wellness")
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()][:10]

if st.button("Score Passage"):
    if not passage.strip():
        st.warning("Please paste a passage."); st.stop()
    if not queries:
        st.warning("Please add 1–10 queries."); st.stop()

    with st.spinner("Embedding & scoring…"):
        gr = gzip_ratio(passage)
        semuniq, tok_count, ctok_count, uniq_count = semantic_uniques_score(passage)
        gzip_adj   = gr / max(1e-9, math.log1p(tok_count))
        gzip_norm  = squash_01(gzip_adj)
        semu_norm  = squash_01(semuniq)
        ov_len, win_scores, sims, q_labels, w_labels = overlap_embed(
            passage, queries, model_name=model_name, win_size=win_size, stride=stride
        )
        final = geometric_mean([gzip_norm, semu_norm, ov_len])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gzip (density, norm)", f"{gzip_norm:.2f}")
    m2.metric("Semantic Uniques (norm)", f"{semu_norm:.2f}")
    m3.metric("Overlap (relevance)",    f"{ov_len:.2f}")
    m4.metric("Content Balance Score",  f"{final:.2f}")

    if tok_count < 25:
        st.info("Very short passages (< 25 tokens) can be unstable.")

    st.markdown("---")
    st.subheader("Annotated Passage (overlap heat)")
    st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>"+render_highlighted(passage, win_scores)+"</div>",
                unsafe_allow_html=True)


    with st.expander("Details"):
        st.write(f"Raw gzip ratio: {gr:.4f}")
        st.write(f"Tokens: {tok_count} | Content tokens: {ctok_count} | Unique content tokens (≥3): {uniq_count}")
        st.write("Window spans (char offsets) with max-overlap score:"); st.write(win_scores)
