import math
import gzip
import io
import re
from typing import List, Tuple
import numpy as np
import streamlit as st

# ---------- Optional NER (tries spaCy; falls back if not installed) ----------
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# ---------- SBERT ----------
from sentence_transformers import SentenceTransformer
_SBERT = None

def get_sbert(model_name: str):
    global _SBERT
    # Cache a single model instance
    if _SBERT is None or getattr(_SBERT, "_name_or_path", "") != model_name:
        _SBERT = SentenceTransformer(model_name)
        _SBERT._name_or_path = model_name  # tag for reuse
    return _SBERT

# ---------- Simple tokenization & helpers ----------
STOP = {
    "the","a","an","and","or","but","if","then","so","as","at","by","for","in","of","on","to","with",
    "is","are","was","were","be","been","being","it","its","this","that","these","those","we","you",
    "your","our","their","from","over","into","out","up","down","about","than","too","very"
}
TOKEN_SPLIT = re.compile(r"[^\w'-]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    tokens = [t for t in TOKEN_SPLIT.split(text.lower()) if t]
    return tokens

def content_tokens(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP]

def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as f:
        f.write(raw)
    compressed_len = len(buf.getvalue())
    return compressed_len / max(1, len(raw))

def semantic_uniques_score(text: str) -> Tuple[float, int, int, int]:
    toks = tokenize(text)
    ctoks = content_tokens(toks)
    uniques = {t for t in ctoks if len(t) >= 3}
    score = (len(uniques) / max(1, len(ctoks))) if ctoks else 0.0
    return score, len(toks), len(ctoks), len(uniques)

def squash_01(x: float) -> float:
    return 1.0 - math.exp(-3.0 * max(0.0, x))

def geometric_mean(vals: List[float]) -> float:
    vals = [max(1e-9, min(1.0, v)) for v in vals]
    return float(np.exp(np.mean(np.log(vals))))

# ---------- Text windowing ----------
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sents(text: str) -> List[Tuple[int, int, str]]:
    sents = []
    idx = 0
    for part in SENT_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        start = text.find(part, idx)
        end = start + len(part)
        sents.append((start, end, part))
        idx = end
    if not sents:
        sents = [(0, len(text), text)]
    return sents

def sliding_windows(sents: List[Tuple[int,int,str]], win_size=3, stride=2):
    windows = []
    for i in range(0, len(sents), stride):
        chunk = sents[i:i+win_size]
        if not chunk:
            continue
        start = chunk[0][0]
        end = chunk[-1][1]
        text = " ".join([c[2] for c in chunk])
        windows.append((start, end, text))
    return windows

# ---------- Embeddings & similarity (SBERT) ----------
def embed_sbert(texts: List[str], model_name: str) -> np.ndarray:
    model = get_sbert(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a and b are expected L2-normalized (we normalize in model.encode)
    return a @ b.T

# ---------- Overlap scoring ----------
def calc_overlap_sbert(passage: str, queries: List[str], model_name="all-MiniLM-L6-v2",
                       win_size=3, stride=2):
    """
    Returns:
      overlap_len (float),
      window_scores: list of (start, end, max_sim_over_all_queries),
      sim_matrix: shape (#queries, #windows),
      query_labels,
      window_labels
    """
    sents = split_sents(passage)
    wins = sliding_windows(sents, win_size=win_size, stride=stride)
    if not wins:
        wins = [(0, len(passage), passage)]

    window_texts = [w[2] for w in wins]
    window_spans = [(w[0], w[1]) for w in wins]

    # Embed windows + queries together for efficiency
    embed_inputs = queries + window_texts
    vecs = embed_sbert(embed_inputs, model_name=model_name)
    q_vecs = vecs[:len(queries)]
    w_vecs = vecs[len(queries):]

    sims = cosine_sim(q_vecs, w_vecs)  # (Q, W)
    per_q_max = np.max(sims, axis=1) if sims.size else np.array([0.0]*len(queries))
    overlap_raw = float(np.mean(per_q_max)) if len(per_q_max) else 0.0

    tokens = len(tokenize(passage))
    length_factor = min(1.0, math.log1p(tokens) / math.log1p(40.0))
    overlap_len = max(0.0, min(1.0, overlap_raw * length_factor))

    per_win_max = np.max(sims, axis=0) if sims.size else np.zeros(len(wins))
    window_scores = [(span[0], span[1], float(v)) for span, v in zip(window_spans, per_win_max)]

    q_labels = [f"Q{i+1}" for i in range(len(queries))]
    w_labels = [f"W{i+1}" for i in range(len(wins))]
    return overlap_len, window_scores, sims, q_labels, w_labels

# ---------- Highlighting ----------
def color_for_score(v: float) -> str:
    v = max(0.0, min(1.0, v))
    base = np.array([255, 255, 255], dtype=float)
    target = np.array([255, 213, 79], dtype=float)  # amber 300
    rgb = (base + v * (target - base)).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def render_highlighted(passage: str, window_scores: List[Tuple[int,int,float]]) -> str:
    if not passage:
        return ""
    scores = np.zeros(len(passage), dtype=float)
    for s, e, v in window_scores:
        s = max(0, min(len(passage), s))
        e = max(0, min(len(passage), e))
        if e > s:
            scores[s:e] = np.maximum(scores[s:e], v)
    html = []
    def bucket(x): return round(x, 2)
    i = 0
    while i < len(passage):
        b = bucket(scores[i])
        j = i + 1
        while j < len(passage) and bucket(scores[j]) == b:
            j += 1
        segment = (passage[i:j]
                   .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        html.append(f"<span style='background:{color_for_score(b)}'>{segment}</span>")
        i = j
    return "".join(html)

# ---------- NER ----------
ENTITY_COLORS = {
    "PERSON": "#e1f5fe",
    "ORG": "#e8f5e9",
    "GPE": "#fff3e0",
    "LOC": "#f3e5f5",
    "PRODUCT": "#fce4ec",
    "EVENT": "#e0f7fa",
}

def extract_entities(text: str):
    if _NLP:
        doc = _NLP(text)
        return [(ent.start_char, ent.end_char, ent.text, ent.label_) for ent in doc.ents]
    ents = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text):
        ents.append((m.start(), m.end(), m.group(0), "PROPN"))
    return ents

def render_entities_layer(passage: str, ents):
    layers = []
    for (s, e, _, label) in ents:
        s = max(0, min(len(passage), s))
        e = max(0, min(len(passage), e))
        if e <= s: 
            continue
        layers.append((s, e, label))
    layers.sort(key=lambda x: (x[0], -x[1]))
    html = []
    i = 0
    while i < len(passage):
        match = None
        for (s, e, lab) in layers:
            if s == i:
                match = (s, e, lab)
                break
        if match:
            s, e, lab = match
            color = ENTITY_COLORS.get(lab, "#eeeeee")
            seg = (passage[s:e]
                   .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
            html.append(
                f"<span style='box-shadow: inset 0 -0.6em 0 {color}; border-bottom:1px solid #999'>{seg}<sup style='font-size:0.7em;color:#666'> {lab}</sup></span>"
            )
            i = e
        else:
            ch = passage[i]
            ch = ch.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html.append(ch)
            i += 1
    return "".join(html)

# ---------- UI ----------
st.set_page_config(page_title="Semantic Overlap & Density Visualizer (SBERT)", layout="wide")
st.title("Semantic Overlap & Density — Visual Annotator (SBERT)")
st.caption("Local embeddings via Sentence-Transformers. Inspired by Duane Forrester’s ‘Semantic Overlap vs Density’.")

with st.sidebar:
    st.header("Settings")
    sbert_model = st.selectbox(
        "SBERT model",
        [
            "all-MiniLM-L6-v2",          # 384-dim, fast, great default
            "all-mpnet-base-v2",         # 768-dim, higher quality, slower
            "multi-qa-MiniLM-L6-cos-v1"  # QA-tuned variant
        ],
        index=0
    )
    win_size = st.slider("Sentence window size", 1, 6, 3)
    stride = st.slider("Window stride", 1, 6, 2)
    st.markdown("---")
    st.write("**Tips**")
    st.write("- Provide 5–10 queries for steadier overlap.")
    st.write("- Very short passages (< 25 tokens) can be unstable.")

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Passage")
    passage = st.text_area("Paste the passage to score:", height=230, placeholder="Paste your content here...")

with colB:
    st.subheader("Queries (one per line, up to 10)")
    raw_queries = st.text_area("Queries:", height=230, placeholder="e.g.\nluxury resort whistler\nski-in ski-out suites\nspa and wellness")
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()][:10]

run = st.button("Score Passage")

if run:
    if not passage.strip():
        st.warning("Please paste a passage.")
        st.stop()
    if not queries:
        st.warning("Please add 1–10 queries.")
        st.stop()

    # Axes 1–2
    gr = gzip_ratio(passage)
    semuniq, tok_count, ctok_count, uniq_count = semantic_uniques_score(passage)
    gzip_adj = gr / max(1e-9, math.log1p(tok_count))
    gzip_norm = squash_01(gzip_adj)
    semuniq_norm = squash_01(semuniq)

    # Axis 3 (SBERT overlap)
    try:
        overlap_len, window_scores, sims, q_labels, w_labels = calc_overlap_sbert(
            passage, queries, model_name=sbert_model, win_size=win_size, stride=stride
        )
    except Exception as e:
        st.error(f"SBERT embedding failed: {e}")
        st.stop()

    # Fused score
    final_score = geometric_mean([gzip_norm, semuniq_norm, overlap_len])

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gzip (density, norm)", f"{gzip_norm:.2f}")
    m2.metric("Semantic Uniques (norm)", f"{semuniq_norm:.2f}")
    m3.metric("Overlap (relevance)", f"{overlap_len:.2f}")
    m4.metric("Content Balance Score", f"{final_score:.2f}")

    if tok_count < 25:
        st.info("Heads-up: very short passages (< 25 tokens) can produce unstable scores.")

    st.markdown("---")

    # Visuals
    st.subheader("Annotated Passage")
    tabs = st.tabs(["Overlap heat", "Entities", "Both layers (stacked)"])

    # Overlap-only
    with tabs[0]:
        html = render_highlighted(passage, window_scores)
        st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>" + html + "</div>", unsafe_allow_html=True)
        st.caption("Darker background indicates stronger semantic overlap with your queries.")

    # Entities-only
    ents = extract_entities(passage)
    with tabs[1]:
        ehtml = render_entities_layer(passage, ents)
        st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>" + ehtml + "</div>", unsafe_allow_html=True)
        if _NLP:
            st.caption("Entities from spaCy. Colors denote types (ORG, GPE, etc.).")
        else:
            st.caption("Simple heuristic entities shown (install spaCy + model for better results).")

    # Combined (stacked approximation)
    with tabs[2]:
        st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>" + render_highlighted(passage, window_scores) + "</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>" + ehtml + "</div>", unsafe_allow_html=True)

    # Heatmap (queries x windows)
    st.subheader("Query × Passage Windows (Similarity Heatmap)")
    if sims.size:
        import pandas as pd
        df = pd.DataFrame(sims, index=q_labels, columns=w_labels)
        st.dataframe(df.style.background_gradient(axis=None), use_container_width=True)
    else:
        st.info("No similarity matrix (empty inputs).")

    # Details
    with st.expander("Details & Debug"):
        st.write(f"Raw gzip ratio: {gr:.4f}")
        st.write(f"Tokens: {tok_count} | Content tokens: {ctok_count} | Unique content tokens (≥3): {uniq_count}")
        st.write("Window spans (character offsets) with max overlap value:")
        st.write(window_scores)

    st.markdown("---")
    # Rewrite assistant (simple, local heuristic)
    st.subheader("Rewrite & Rescore (keep it on-brand)")
    rewrite_goal = st.slider("Target Content Balance Score", 0.0, 1.0, 0.5, 0.05)
    prompt = st.text_area(
        "Rewrite brief (optional)",
        placeholder="E.g., keep luxury tone, concise, mention ski-in/ski-out and spa; avoid clichés; no emojis."
    )
    if st.button("Suggest rewrite (local heuristic)"):
        lines = [l.strip() for l in passage.splitlines() if l.strip()]
        trimmed = []
        for l in lines:
            l2 = re.sub(r"\b(very|really|just|quite|simply|actually|basically)\b", "", l, flags=re.I)
            l2 = re.sub(r"\s{2,}", " ", l2).strip()
            if len(l2.split()) > 4:
                trimmed.append(l2)
        # soft query variants
        soft_q = []
        for q in queries:
            q2 = q.lower()
            q2 = q2.replace("luxury", "high-end").replace("spa", "wellness")
            if q2 not in soft_q:
                soft_q.append(q2)
        add = " ".join({sq for sq in soft_q})
        candidate = " ".join(trimmed)
        if add:
            candidate = candidate + ". " + add.capitalize() + "."
        st.text_area("Draft rewrite", candidate, height=180)
        st.info("Tip: paste this draft back into the Passage box and click **Score Passage** to iterate toward your target.")
