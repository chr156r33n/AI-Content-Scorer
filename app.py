import math, gzip, io, re, html
from typing import List, Tuple
from collections import Counter, defaultdict
import numpy as np
import streamlit as st

# ---- Fast, torch-free embeddings ----
from fastembed import TextEmbedding

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    # First call downloads a small ONNX (~50–90MB) into HF cache; later runs are instant
    return TextEmbedding(model_name=model_name)

# ----------------- Text utils -----------------
STOP = {"the","a","an","and","or","but","if","then","so","as","at","by","for","in","of","on","to","with",
        "is","are","was","were","be","been","being","it","its","this","that","these","those","we","you",
        "your","our","their","from","over","into","out","up","down","about","than","too","very"}
TOKEN_SPLIT = re.compile(r"[^\w'-]+", re.UNICODE)
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")
WORD_RE     = re.compile(r"\b\w+\b")

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
    counts = Counter(ctoks)
    uniques = {w for w,c in counts.items() if c == 1 and len(w) >= 3}
    score = (len(uniques) / max(1, len(ctoks))) if ctoks else 0.0
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

# ----------------- Embeddings -----------------
def embed_fastembed(texts: List[str], model_name="BAAI/bge-small-en-v1.5") -> np.ndarray:
    model = get_embedder(model_name)
    vecs = np.stack(list(model.embed(texts))).astype(np.float32)  # generator -> array
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms  # L2-normalized

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # normalized dot = cosine

def overlap_embed(passage: str, queries: List[str], model_name="BAAI/bge-small-en-v1.5",
                  win_size=3, stride=2):
    """
    Returns:
      overlap_len (float),
      window_scores: list of (start, end, max_sim, contributing_queries[(q_idx, query, score), ...]),
      sims (Q x W),
      q_labels, w_labels
    """
    sents = split_sents(passage)
    wins = sliding_windows(sents, win_size=win_size, stride=stride) or [(0, len(passage), passage)]
    w_texts = [w[2] for w in wins]; w_spans = [(w[0], w[1]) for w in wins]

    vecs = embed_fastembed(queries + w_texts, model_name=model_name)
    q_vecs, w_vecs = vecs[:len(queries)], vecs[len(queries):]
    sims = cosine_sim(q_vecs, w_vecs) if (len(queries) and len(w_texts)) else np.zeros((0,0))

    per_q_max = np.max(sims, axis=1) if sims.size else np.array([0.0]*len(queries))
    overlap_raw = float(np.mean(per_q_max)) if len(per_q_max) else 0.0

    tokens = len(tokenize(passage))
    length_factor = min(1.0, math.log1p(tokens) / math.log1p(40.0))
    overlap_len = max(0.0, min(1.0, overlap_raw * length_factor))

    per_win_max = np.max(sims, axis=0) if sims.size else np.zeros(len(wins))

    window_scores = []
    for i, (span, max_score) in enumerate(zip(w_spans, per_win_max)):
        contributing = []
        if sims.size:
            for j, q in enumerate(queries):
                if i < sims.shape[1]:
                    qscore = sims[j, i]
                    # include queries within 80% of window's max
                    if max_score > 0 and qscore >= max_score * 0.8:
                        contributing.append((j, q, float(qscore)))
        contributing.sort(key=lambda x: x[2], reverse=True)
        window_scores.append((span[0], span[1], float(max_score), contributing))

    q_labels = [f"Q{i+1}" for i in range(len(queries))]
    w_labels = [f"W{i+1}" for i in range(len(wins))]
    return overlap_len, window_scores, sims, q_labels, w_labels

# ----------------- Visual rendering -----------------
def color_for_score(v: float) -> str:
    """Dotted red border; thickness scales with score."""
    v = max(0.0, min(1.0, v))
    width = max(1, int(round(v * 3)))   # 1–3px
    alpha = max(0.35, min(1.0, v))
    return f"border: {width}px dotted rgba(255,0,0,{alpha:.2f}); padding: 2px 3px; border-radius: 2px;"

def get_unique_words(text: str) -> set:
    """Unique content words: non-stop, >=3 chars, appear exactly once (lowercased)."""
    toks = tokenize(text)
    ctoks = content_tokens(toks)
    counts = Counter(ctoks)
    return {w for w,c in counts.items() if c == 1 and len(w) >= 3}

def get_unique_word_spans(passage: str, unique_words: set) -> list[tuple[int,int]]:
    """Return (start,end) spans for unique words (case-insensitive) in the original text."""
    spans = []
    for m in WORD_RE.finditer(passage):
        if m.group(0).lower() in unique_words:
            spans.append((m.start(), m.end()))
    return spans

def render_highlighted(passage: str, window_scores: List[Tuple[int,int,float,list]]) -> str:
    """
    Event-based renderer:
      - Red dotted borders around each window span (supports nesting/overlap)
      - Green bold for unique words (applies everywhere, including inside windows)
    """
    if not passage:
        return ""

    unique_words = get_unique_words(passage)
    uniq_spans    = get_unique_word_spans(passage, unique_words)

    # Build event maps: opens/closes at character positions
    opens  = defaultdict(list)   # pos -> list of events
    closes = defaultdict(list)   # pos -> list of events

    # Window events (open outermost by higher score first)
    for (start, end, score, contrib) in window_scores:
        start = max(0, min(len(passage), start))
        end   = max(0, min(len(passage), end))
        if end <= start:
            continue
        opens[start].append({"type": "win", "score": float(score), "contrib": contrib})
        closes[end].append({"type": "win", "score": float(score), "contrib": contrib})

    # Unique word events
    for (s, e) in uniq_spans:
        opens[s].append({"type": "uniq"})
        closes[e].append({"type": "uniq"})

    boundaries = sorted(set([0, len(passage)] + list(opens.keys()) + list(closes.keys())))
    out = []
    pos = 0
    stack = []  # each item: {"type": "win"/"uniq", ...}

    def open_win(score: float):
        out.append(f"<span style='{color_for_score(score)}'>")

    def close_win(score: float, contrib):
        parts = [f"Q{qi+1}:{qs:.2f}" for (qi, _q, qs) in (contrib[:2] if contrib else [])]
        badge = f" ({score:.2f}" + (f" | {' | '.join(parts)}" if parts else "") + ")"
        out.append(f"<sup style='font-size:0.7em; color:#666'>{html.escape(badge)}</sup>")
        out.append("</span>")

    def open_uniq():
        out.append("<span style='color:green; font-weight:600'>")

    def close_uniq():
        out.append("</span>")

    for b in boundaries:
        # Emit plain text between last boundary and this boundary
        if b > pos:
            out.append(html.escape(passage[pos:b]))
            pos = b

        # Close events at boundary: close uniques first (inner), then windows (outer)
        if b in closes:
            # close unique spans
            for _ in [e for e in closes[b] if e["type"] == "uniq"]:
                for i in range(len(stack)-1, -1, -1):
                    if stack[i]["type"] == "uniq":
                        stack.pop(i); close_uniq(); break
            # close windows
            for _ in [e for e in closes[b] if e["type"] == "win"]:
                for i in range(len(stack)-1, -1, -1):
                    if stack[i]["type"] == "win":
                        win = stack.pop(i); close_win(win["score"], win.get("contrib")); break

        # Open events at boundary: open windows first (outer), then uniques (inner)
        if b in opens:
            for ev in sorted([e for e in opens[b] if e["type"] == "win"],
                             key=lambda d: d["score"], reverse=True):
                open_win(ev["score"]); stack.append(ev)
            for ev in [e for e in opens[b] if e["type"] == "uniq"]:
                open_uniq(); stack.append(ev)

    # Safety: close any unclosed tags
    while stack:
        ev = stack.pop()
        if ev["type"] == "uniq":
            close_uniq()
        else:
            close_win(ev["score"], ev.get("contrib"))

    return "".join(out)

# ----------------- UI -----------------
st.set_page_config(page_title="Semantic Overlap & Density (FastEmbed)", layout="wide")
st.title("Semantic Overlap & Density — FastEmbed (no Torch)")
st.caption("CPU-only ONNX embeddings with clean visual annotations (windows + unique words).")

with st.sidebar:
    model_name = st.selectbox(
        "Embedding model",
        ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"],
        index=0,
    )
    win_size = st.slider("Sentence window size", 1, 6, 3)
    stride   = st.slider("Window stride", 1, 6, 2)

colA, colB = st.columns([1,1])
with colA:
    passage = st.text_area("Passage", height=230, placeholder="Paste your content here…")
with colB:
    raw_queries = st.text_area(
        "Queries (one per line, up to 10)",
        height=230,
        placeholder="e.g.\nluxury resort whistler\nski-in ski-out suites\nspa and wellness"
    )
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()][:10]

if st.button("Score Passage"):
    if not passage.strip():
        st.warning("Please paste a passage."); st.stop()
    if not queries:
        st.warning("Please add 1–10 queries."); st.stop()

    with st.spinner("Embedding & scoring…"):
        gr = gzip_ratio(passage)
        semuniq, tok_count, ctok_count, uniq_count = semantic_uniques_score(passage)
        gzip_adj  = gr / max(1e-9, math.log1p(tok_count))
        gzip_norm = squash_01(gzip_adj)
        semu_norm = squash_01(semuniq)
        ov_len, win_scores, sims, q_labels, w_labels = overlap_embed(
            passage, queries, model_name=model_name, win_size=win_size, stride=stride
        )
        final = geometric_mean([gzip_norm, semu_norm, ov_len])

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gzip (density, norm)", f"{gzip_norm:.2f}")
    m2.metric("Semantic Uniques (norm)", f"{semu_norm:.2f}")
    m3.metric("Overlap (relevance)",    f"{ov_len:.2f}")
    m4.metric("Content Balance Score",  f"{final:.2f}")

    if tok_count < 25:
        st.info("Very short passages (< 25 tokens) can be unstable.")

    st.markdown("---")
    st.subheader("Annotated Passage")
    st.markdown(
        "<div style='line-height:1.8; font-size:1.05rem;'>"
        + render_highlighted(passage, win_scores)
        + "</div>",
        unsafe_allow_html=True
    )

    with st.expander("Details"):
        st.write(f"Raw gzip ratio: {gr:.4f}")
        st.write(f"Tokens: {tok_count} | Content tokens: {ctok_count} | Unique content tokens (≥3): {uniq_count}")
        st.write("Window spans (char offsets) with max-overlap score & top queries:")
        st.write(win_scores)
