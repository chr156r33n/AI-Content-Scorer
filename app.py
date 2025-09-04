import math, gzip, io, re, html
from typing import List, Tuple
from collections import Counter, defaultdict
import numpy as np
import streamlit as st

# ---------------- Config ----------------
MIN_WINDOW_BORDER_SCORE = 0.0  # border visible only if score >= this; badges always shown

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
      window_scores: list of (start, end, max_sim, contributing_queries[(q_idx, query, score), ...], render_flag),
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
        render_flag = float(max_score) >= MIN_WINDOW_BORDER_SCORE
        window_scores.append((span[0], span[1], float(max_score), contributing, render_flag))

    q_labels = [f"Q{i+1}" for i in range(len(queries))]
    w_labels = [f"W{i+1}" for i in range(len(wins))]
    return overlap_len, window_scores, sims, q_labels, w_labels

# ----------------- Visual rendering -----------------
WINDOW_PALETTE = [
    "#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa",
    "#00897b", "#6d4c41", "#3949ab", "#c0ca33", "#f4511e"
]

def color_for_window(win_idx: int, score: float) -> str:
    """Dotted border style per-window with thickness by score."""
    c = WINDOW_PALETTE[win_idx % len(WINDOW_PALETTE)]
    v = max(0.0, min(1.0, score))
    width = max(1, int(round(v * 3)))   # 1–3px
    alpha = max(0.35, min(1.0, v))
    return (
        f"border: {width}px dotted {c};"
        f"padding: 2px 3px; border-radius: 2px;"
        f"opacity: {alpha:.2f};"
    )

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

def render_highlighted(passage: str, window_scores: List[Tuple[int,int,float,list,bool]]) -> str:
    """
    ID-aware renderer (handles overlapping & crossing windows):
      - Each window has a unique color (stable by index in window_scores).
      - Tiny score badge at open; detailed badge (+top queries) at close.
      - Green bold for unique words everywhere (inside/outside windows).
      - Borders shown only if score >= MIN_WINDOW_BORDER_SCORE; badges always shown.
    """
    if not passage:
        return ""

    unique_words = get_unique_words(passage)
    uniq_spans    = get_unique_word_spans(passage, unique_words)

    # Build events with stable window IDs
    opens, closes = defaultdict(list), defaultdict(list)
    win_meta = {}  # wid -> dict(score, contrib, render, style)

    for wid, ws in enumerate(window_scores):
        start, end, score, contrib, render_flag = ws
        start = max(0, min(len(passage), int(start)))
        end   = max(0, min(len(passage), int(end)))
        if end <= start:
            continue
        style = color_for_window(wid, float(score)) if render_flag else "padding:0;"
        win_meta[wid] = {"score": float(score), "contrib": contrib, "render": bool(render_flag), "style": style}
        opens[start].append({"type":"win", "id": wid})
        closes[end].append({"type":"win", "id": wid})

    # Unique word events
    for (s, e) in uniq_spans:
        opens[s].append({"type":"uniq"})
        closes[e].append({"type":"uniq"})

    boundaries = sorted(set([0, len(passage)] + list(opens.keys()) + list(closes.keys())))
    out, pos = [], 0
    active_win_ids = []   # track exact active windows (order of opening)
    uniq_depth = 0

    def open_win(wid: int):
        meta = win_meta[wid]
        score = meta["score"]
        out.append(f"<sup style='font-size:0.65em; color:#888'>{score:.2f}</sup>")
        out.append(f"<span data-win='{wid}' style='{meta['style']}'>")
        active_win_ids.append(wid)

    def close_win(wid: int):
        if wid in active_win_ids:
            meta = win_meta[wid]
            score, contrib = meta["score"], meta["contrib"]
            parts = [f"Q{qi+1}:{qs:.2f}" for (qi, _q, qs) in (contrib[:2] if contrib else [])]
            badge = f" ({score:.2f}" + (f" | {' | '.join(parts)}" if parts else "") + ")"
            out.append(f"<sup style='font-size:0.7em; color:#666'>{html.escape(badge)}</sup>")
            out.append("</span>")
            active_win_ids.remove(wid)

    def open_uniq():
        nonlocal uniq_depth
        out.append("<span style='color:green; font-weight:600'>")
        uniq_depth += 1

    def close_uniq():
        nonlocal uniq_depth
        if uniq_depth > 0:
            uniq_depth -= 1
            out.append("</span>")

    for b in boundaries:
        # Emit plain text between last boundary and this boundary
        if b > pos:
            out.append(html.escape(passage[pos:b]))
            pos = b

        # Close events at this boundary
        if b in closes:
            # Close unique spans first (innermost)
            for _ in [e for e in closes[b] if e["type"] == "uniq"]:
                close_uniq()
            # Close each window that *ends here* by exact id (handles crossing)
            for ev in [e for e in closes[b] if e["type"] == "win"]:
                close_win(ev["id"])

        # Open events at this boundary
        if b in opens:
            # Open windows first (outer), sorted by score desc to put stronger outside
            for ev in sorted([e for e in opens[b] if e["type"] == "win"],
                             key=lambda e: win_meta[e["id"]]["score"],
                             reverse=True):
                open_win(ev["id"])
            # Then open unique spans (inner)
            for ev in [e for e in opens[b] if e["type"] == "uniq"]:
                open_uniq()

    # Safety close (should be empty)
    for wid in list(active_win_ids)[::-1]:
        close_win(wid)
    while uniq_depth > 0:
        close_uniq()

    return "".join(out)

# ----------------- UI -----------------
st.set_page_config(page_title="Semantic Overlap & Density (FastEmbed)", layout="wide")
st.title("Semantic Overlap & Density — FastEmbed (no Torch)")
st.caption("CPU-only ONNX embeddings with clear annotations: per-window colors, unique-word highlights, legend & summary.")

with st.sidebar:
    model_name = st.selectbox(
        "Embedding model",
        ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"],
        index=0,
    )
    win_size = st.slider("Sentence window size", 1, 6, 3)
    stride   = st.slider("Window stride", 1, 6, 2)
    border_thresh = st.slider("Window border threshold", 0.0, 1.0, float(MIN_WINDOW_BORDER_SCORE), 0.05,
                              help="Borders show only if score ≥ threshold. Badges always show.")
    MIN_WINDOW_BORDER_SCORE = border_thresh  # reflect live

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
    m2.metric("Semantic Uniques (norm)", f"{semun_norm:.2f}")
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

    # ---- Legend & written summary ----
    st.markdown("### Legend")
    legend_cols = st.columns(min(4, max(1, len(win_scores))))
    for i, ws in enumerate(win_scores):
        _, _, score, contrib, render_flag = ws
        color = WINDOW_PALETTE[i % len(WINDOW_PALETTE)]
        with legend_cols[i % len(legend_cols)]:
            st.markdown(
                f"<div style='display:inline-block;width:14px;height:14px;background:{color};"
                f"border-radius:2px;margin-right:8px;vertical-align:middle;'></div>"
                f"<span><b>W{i+1}</b> (score {score:.2f})</span>",
                unsafe_allow_html=True
            )

    st.markdown("### Query Key")
    st.markdown(", ".join([f"**Q{i+1}**: {q}" for i, q in enumerate(queries)]) or "_No queries_")

    st.markdown("### Window Summary")
    for i, ws in enumerate(win_scores):
        start, end, score, contrib, render_flag = ws
        top = ", ".join([f"Q{qi+1} {qs:.2f}" for (qi, _q, qs) in (contrib[:3] if contrib else [])]) or "—"
        st.write(f"**W{i+1}** [{start}:{end}] • score **{score:.2f}** • top queries: {top}")

    with st.expander("Details"):
        st.write(f"Raw gzip ratio: {gr:.4f}")
        st.write(f"Tokens: {tok_count} | Content tokens: {ctok_count} | Unique content tokens (≥3): {uniq_count}")
        st.write("Window tuples (start, end, score, top queries (idx, text, score), render_flag):")
        st.write(win_scores)
