import math, gzip, io, re, html, hashlib, json, textwrap, difflib
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np
import streamlit as st
import requests  # used ONLY when user provides an API key

# ---------------- Config ----------------
MIN_WINDOW_BORDER_SCORE = 0.0  # border visible only if score >= this; badges always shown
MAX_QUERIES = 10
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
GPT35_DEFAULT = "gpt-3.5-turbo-0125"   # safe default; change if needed

# ---- Fast, torch-free embeddings ----
from fastembed import TextEmbedding

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    return TextEmbedding(model_name=model_name)

# ----------------- Text utils -----------------
STOP = {"the","a","an","and","or","but","if","then","so","as","at","by","for","in","of","on","to","with",
        "is","are","was","were","be","been","being","it","its","this","that","these","those","we","you",
        "your","our","their","from","over","into","out","up","down","about","than","too","very"}
TOKEN_SPLIT = re.compile(r"[^\w'-]+", re.UNICODE)
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")
WORD_RE     = re.compile(r"\b\w+\b", re.UNICODE)

FILLER_WORDS = re.compile(r"\b(?:very|really|just|quite|simply|actually|basically|kind of|sort of|maybe|perhaps)\b", re.I)
PASSIVE_HINT = re.compile(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", re.I)

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
    vecs = np.stack(list(model.embed(texts))).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def overlap_embed(passage: str, queries: List[str], model_name="BAAI/bge-small-en-v1.5",
                  win_size=3, stride=2):
    """
    Returns:
      overlap_len (float),
      window_scores: list of (start, end, max_sim, contrib[(q_idx, query, score), ...], render_flag),
      sims (Q x W),
      q_labels, w_labels,
      wins (list of (start,end,text)) for per-query stripes
    """
    sents = split_sents(passage)
    wins = sliding_windows(sents, win_size=win_size, stride=stride) or [(0, len(passage), passage)]
    w_texts = [w[2] for w in wins]; w_spans = [(w[0], w[1]) for w in wins]

    vecs = embed_fastembed(queries + w_texts, model_name=model_name) if queries else np.zeros((0,0))
    q_vecs = vecs[:len(queries)] if queries else np.zeros((0,0))
    w_vecs = vecs[len(queries):] if queries else np.zeros((len(wins),0))
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
                    if max_score > 0 and qscore >= max_score * 0.8:
                        contributing.append((j, q, float(qscore)))
        contributing.sort(key=lambda x: x[2], reverse=True)
        render_flag = float(max_score) >= MIN_WINDOW_BORDER_SCORE
        window_scores.append((span[0], span[1], float(max_score), contributing, render_flag))

    q_labels = [f"Q{i+1}" for i in range(len(queries))]
    w_labels = [f"W{i+1}" for i in range(len(wins))]
    return overlap_len, window_scores, sims, q_labels, w_labels, wins

# ----------------- Visual rendering (windows + uniques) -----------------
WINDOW_PALETTE = [
    "#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa",
    "#00897b", "#6d4c41", "#3949ab", "#c0ca33", "#f4511e"
]
QUERY_PALETTE = [
    "#e91e63", "#3f51b5", "#009688", "#ff5722", "#673ab7",
    "#2196f3", "#4caf50", "#ff9800", "#ffeb3b", "#00bcd4"
]

def color_for_window(win_idx: int, score: float) -> str:
    c = WINDOW_PALETTE[win_idx % len(WINDOW_PALETTE)]
    v = max(0.0, min(1.0, score))
    width = max(1, int(round(v * 3)))   # 1–3px
    alpha = max(0.35, min(1.0, v))
    return f"border: {width}px dotted {c}; padding: 2px 3px; border-radius: 2px; opacity: {alpha:.2f};"

def get_unique_words(text: str) -> set:
    toks = tokenize(text)
    ctoks = content_tokens(toks)
    counts = Counter(ctoks)
    return {w for w,c in counts.items() if c == 1 and len(w) >= 3}

def get_unique_word_spans(passage: str, unique_words: set) -> list[tuple[int,int]]:
    spans = []
    for m in WORD_RE.finditer(passage):
        if m.group(0).lower() in unique_words:
            spans.append((m.start(), m.end()))
    return spans

def build_query_charmap(passage: str, wins: List[Tuple[int,int,str]], sims: np.ndarray, q_count: int) -> np.ndarray:
    """
    Returns (q_count, len(passage)) map where cell is similarity assigned from its window span.
    If a char belongs to multiple windows, keep the max per query.
    """
    L = len(passage)
    cmap = np.zeros((q_count, L), dtype=np.float32)
    if sims.size == 0: return cmap
    for wi, (s, e, _t) in enumerate(wins):
        s = max(0, min(L, s)); e = max(0, min(L, e))
        if e <= s: continue
        for qi in range(q_count):
            val = float(sims[qi, wi])
            if val <= 0: continue
            cmap[qi, s:e] = np.maximum(cmap[qi, s:e], val)
    return cmap

def render_passage(
    passage: str,
    window_scores: List[Tuple[int,int,float,list,bool]],
    wins: List[Tuple[int,int,str]],
    sims: np.ndarray,
    queries: List[str],
    show_per_query_stripes: bool,
    show_filler_flags: bool,
    overstuff_threshold: int,
    show_windows: bool = True
) -> str:
    """ID-aware renderer with optional per-query stripes and filler/over-stuffing overlays."""
    if not passage: return ""
    unique_words = get_unique_words(passage)
    uniq_spans = get_unique_word_spans(passage, unique_words)

    opens, closes = defaultdict(list), defaultdict(list)
    win_meta = {}

    for wid, ws in enumerate(window_scores):
        start, end, score, contrib, render_flag = ws
        start = max(0, min(len(passage), int(start)))
        end   = max(0, min(len(passage), int(end)))
        if end <= start: continue
        if show_windows:
            style = color_for_window(wid, float(score)) if render_flag else "padding:0;"
            win_meta[wid] = {"score": float(score), "contrib": contrib, "render": bool(render_flag), "style": style}
            opens[start].append({"type":"win", "id": wid})
            closes[end].append({"type":"win", "id": wid})
        else:
            win_meta[wid] = {"score": float(score), "contrib": contrib, "render": False, "style": "padding:0;"}

    for (s, e) in uniq_spans:
        opens[s].append({"type":"uniq"})
        closes[e].append({"type":"uniq"})

    filler_spans = []
    if show_filler_flags:
        for m in FILLER_WORDS.finditer(passage):
            filler_spans.append((m.start(), m.end()))
        for (s, e) in filler_spans:
            opens[s].append({"type":"filler"})
            closes[e].append({"type":"filler"})

    over_spans = []
    if overstuff_threshold > 0 and queries:
        low_queries = sorted(queries, key=len, reverse=True)
        counts = Counter()
        for q in low_queries:
            if not q.strip(): continue
            pattern = re.compile(rf"(?i)\b{re.escape(q)}\b")
            counts[q] = len(list(pattern.finditer(passage)))
        for q in low_queries:
            if counts[q] > overstuff_threshold:
                pattern = re.compile(rf"(?i)\b{re.escape(q)}\b")
                idx = 0
                for m in pattern.finditer(passage):
                    idx += 1
                    if idx > overstuff_threshold:
                        over_spans.append((m.start(), m.end()))
        for (s, e) in over_spans:
            opens[s].append({"type":"over"})
            closes[e].append({"type":"over"})

    q_top = None
    if show_per_query_stripes and len(queries):
        q_cmap = build_query_charmap(passage, wins, sims, len(queries))
        if q_cmap.size:
            q_top = np.argmax(q_cmap, axis=0)
            q_max = np.max(q_cmap, axis=0)
            q_top[q_max <= 0] = -1

    boundaries = sorted(set([0, len(passage)] + list(opens.keys()) + list(closes.keys())))
    out, pos = [], 0
    active_win_ids = []
    uniq_depth = 0
    filler_depth = 0
    over_depth = 0

    def open_win(wid: int):
        meta = win_meta[wid]; score = meta["score"]
        out.append(f"<sup style='font-size:0.65em; color:#888'>{score:.2f}</sup>")
        out.append(f"<span data-win='{wid}' style='{meta['style']}'>")
        active_win_ids.append(wid)

    def close_win(wid: int):
        if wid in active_win_ids:
            meta = win_meta[wid]; score, contrib = meta["score"], meta["contrib"]
            parts = [f"Q{qi+1}:{qs:.2f}" for (qi, _q, qs) in (contrib[:2] if contrib else [])]
            badge = f" ({score:.2f}" + (f" | {' | '.join(parts)}" if parts else "") + ")"
            out.append(f"<sup style='font-size:0.7em; color:#666'>{html.escape(badge)}</sup>")
            out.append("</span>")
            active_win_ids.remove(wid)

    def open_uniq():  out.append("<span style='color:green; font-weight:600'>")
    def close_uniq(): out.append("</span>")

    def open_filler():
        out.append("<span style='border-bottom:2px dotted rgba(120,120,120,0.75)'>")
    def close_filler():
        out.append("</span>")

    def open_over():
        out.append("<span style='border-bottom:2px dashed rgba(200,0,0,0.8)'>")
    def close_over():
        out.append("</span>")

    for b in boundaries:
        if b > pos:
            seg = passage[pos:b]
            seg_html = html.escape(seg)

            if q_top is not None:
                colored = []
                i = 0
                while i < len(seg):
                    gidx = pos + i
                    qid = q_top[gidx] if 0 <= gidx < len(q_top) else -1
                    j = i + 1
                    while j < len(seg) and (pos + j) < len(q_top) and q_top[pos + j] == qid:
                        j += 1
                    chunk = html.escape(seg[i:j])
                    if qid >= 0:
                        color = QUERY_PALETTE[qid % len(QUERY_PALETTE)]
                        chunk = f"<span style='background: {color}15; border-bottom: 3px solid {color}; border-radius: 2px; padding: 1px 2px;'>{chunk}</span>"
                    colored.append(chunk)
                    i = j
                seg_html = "".join(colored)

            out.append(seg_html)
            pos = b

        if b in closes:
            for _ in [e for e in closes[b] if e["type"] == "uniq"]:
                close_uniq(); uniq_depth = max(0, uniq_depth-1)
            for _ in [e for e in closes[b] if e["type"] == "filler"]:
                close_filler(); filler_depth = max(0, filler_depth-1)
            for _ in [e for e in closes[b] if e["type"] == "over"]:
                close_over(); over_depth = max(0, over_depth-1)
            for ev in [e for e in closes[b] if e["type"] == "win"]:
                close_win(ev["id"])

        if b in opens:
            for ev in sorted([e for e in opens[b] if e["type"] == "win"],
                             key=lambda e: win_meta[e["id"]]["score"], reverse=True):
                open_win(ev["id"])
            for ev in [e for e in opens[b] if e["type"] == "uniq"]:
                open_uniq(); uniq_depth += 1
            for ev in [e for e in opens[b] if e["type"] == "filler"]:
                open_filler(); filler_depth += 1
            for ev in [e for e in opens[b] if e["type"] == "over"]:
                open_over(); over_depth += 1

    for wid in list(active_win_ids)[::-1]:
        close_win(wid)
    while uniq_depth > 0: close_uniq(); uniq_depth -= 1
    while filler_depth > 0: close_filler(); filler_depth -= 1
    while over_depth > 0: close_over(); over_depth -= 1

    return "".join(out)

# ----------------- Sentence chips & repetition meter -----------------
def sentence_metrics(sent_text: str, queries: List[str], model_name: str) -> Dict[str, float]:
    gr = gzip_ratio(sent_text)
    toks = tokenize(sent_text)
    semuniq, tok_count, ctok_count, uniq_count = semantic_uniques_score(sent_text)
    gzip_adj = gr / max(1e-9, math.log1p(tok_count))
    gzip_norm = squash_01(gzip_adj)
    semu_norm = squash_01(semuniq)

    ov = 0.0
    if queries:
        vecs = embed_fastembed(queries + [sent_text], model_name=model_name)
        qv, sv = vecs[:-1], vecs[-1:]
        sims = (qv @ sv.T).flatten()
        if sims.size:
            ov = float(np.mean(sims))
        length_factor = min(1.0, math.log1p(len(toks)) / math.log1p(20.0))
        ov = max(0.0, min(1.0, ov * length_factor))

    return {"density": gzip_norm, "uniques": semu_norm, "overlap": ov}

def top_ngrams(text: str, n: int = 10) -> Dict[str, int]:
    toks = [t for t in tokenize(text) if len(t) >= 3]
    unis = Counter(toks)
    bigrams = Counter([" ".join(pair) for pair in zip(toks, toks[1:])])
    combined = unis.most_common(n) + bigrams.most_common(n)
    agg = {}
    for k, v in combined:
        agg[k] = max(v, agg.get(k, 0))
    return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:n])

SYNONYM_HINTS = {
    "very": ["extremely", "highly", "especially"],
    "really": ["truly", "genuinely"],
    "just": ["simply", "only"],
    "luxury": ["high-end", "upscale", "premium"],
    "spa": ["wellness", "therapies", "treatments"],
    "hotel": ["property", "retreat", "stay"],
}

def synonym_nudges(terms: List[str]) -> Dict[str, List[str]]:
    hints = {}
    for t in terms:
        base = t.lower()
        if base in SYNONYM_HINTS:
            hints[t] = SYNONYM_HINTS[base]
        elif len(base) > 6:
            hints[t] = [base.rstrip("s"), base + "s"]
    return hints

# ---------- ACTIONABLE EDIT PLAN HELPERS ----------
def words_per_sentence(s: str) -> int:
    return max(1, len([t for t in TOKEN_SPLIT.split(s) if t.strip()]))

def find_low_coverage_gaps(passage: str, wins, sims: np.ndarray, queries: List[str], threshold: float = 0.15):
    if sims.size == 0 or not queries:
        return []
    L = len(passage)
    qmax = np.zeros(L, dtype=np.float32)
    for wi, (s, e, _t) in enumerate(wins):
        s, e = max(0, s), min(L, e)
        if e <= s: continue
        wmax = float(np.max(sims[:, wi])) if sims.size else 0.0
        if wmax <= 0: continue
        qmax[s:e] = np.maximum(qmax[s:e], wmax)
    gaps, i = [], 0
    while i < L:
        if qmax[i] < threshold and not passage[i].isspace():
            j = i + 1
            while j < L and qmax[j] < threshold:
                j += 1
            for (ss, ee, txt) in split_sents(passage):
                if ss <= i < ee:
                    gaps.append((ss, ee))
                    break
            i = j
        else:
            i += 1
    dedup, last = [], (-1, -1)
    for g in gaps:
        if g != last:
            dedup.append(g); last = g
    return dedup[:5]

def query_priorities(queries: List[str], user_weights: Dict[int, float] | None):
    weights = {i: 1.0 for i in range(len(queries))}
    if user_weights:
        weights.update(user_weights)
    return weights

def build_edit_plan(passage: str, queries: List[str], sims: np.ndarray, wins, window_scores, priorities: Dict[int, float], filler_count_limit: int = 2):
    plan = {"additions": [], "trims": [], "rephrases": [], "structure": []}

    gaps = find_low_coverage_gaps(passage, wins, sims, queries, threshold=0.15)
    if sims.size and queries:
        q_order = sorted(range(len(queries)), key=lambda i: -priorities.get(i, 1.0))
        for (ss, ee) in gaps:
            if not q_order: break
            qi = q_order[0]
            plan["additions"].append({
                "where": (ss, ee),
                "hint": f"After this sentence, add a concrete line addressing **Q{qi+1} – {queries[qi]}** (use a specific fact, benefit, or example)."
            })

    for qi, q in enumerate(queries):
        if not q.strip(): continue
        patt = re.compile(rf"(?i)\b{re.escape(q)}\b")
        hits = list(patt.finditer(passage))
        if len(hits) > filler_count_limit:
            for m in hits[filler_count_limit:]:
                plan["trims"].append({"where": (m.start(), m.end()), "hint": f"Replace exact **{q}** with an on-brand variant or pronoun (Q{qi+1})."})

    for m in FILLER_WORDS.finditer(passage):
        plan["trims"].append({"where": (m.start(), m.end()), "hint": "Remove hedge/filler for tighter prose."})
    for (ss, ee, txt) in split_sents(passage):
        if words_per_sentence(txt) > 26:
            plan["rephrases"].append({"where": (ss, ee), "hint": "Split this long sentence (aim ≤ 22 words)."})
        if PASSIVE_HINT.search(txt):
            plan["rephrases"].append({"where": (ss, ee), "hint": "Prefer active voice: ‘We do X’ vs ‘X is done’. "})

    low = [(i, ws) for i, ws in enumerate(window_scores) if ws[2] < 0.2]
    high = [(i, ws) for i, ws in enumerate(window_scores) if ws[2] > 0.5]
    if low:
        i, (s, e, sc, _, _) = low[0]
        plan["structure"].append({"where": (s, e), "hint": f"Section W{i+1} is thin (score {sc:.2f}). Add a specific claim, data point, or example tied to a priority query."})
    if high:
        i, (s, e, sc, contrib, _) = max(high, key=lambda t: t[1][2])
        topq = ", ".join([f"Q{qi+1}" for (qi, _q, _qs) in contrib[:2]]) or "Q1"
        plan["structure"].append({"where": (s, e), "hint": f"Strong section W{i+1} (score {sc:.2f}) linked to {topq}. Consider moving key lines from here earlier."})

    return plan

def render_edit_plan(plan, passage: str):
    def show_item(tag, item):
        s, e = item["where"]
        excerpt = html.escape(passage[s:e][:140]).replace("\n", " ")
        st.markdown(f"- **{tag}**: {item['hint']}  \n  <span style='color:#666'>…{excerpt}…</span>", unsafe_allow_html=True)

    if plan["additions"]:
        st.markdown("**Add**")
        for it in plan["additions"]:
            show_item("Add", it)
    if plan["trims"]:
        st.markdown("**Trim / Replace**")
        for it in plan["trims"][:6]:
            show_item("Trim", it)
    if plan["rephrases"]:
        st.markdown("**Rephrase**")
        for it in plan["rephrases"][:6]:
            show_item("Rewrite", it)
    if plan["structure"]:
        st.markdown("**Structure**")
        for it in plan["structure"]:
            show_item("Structure", it)

# ----------------- OpenAI GPT-3.5 integration (optional) -----------------
def build_llm_prompt(
    passage: str,
    queries: List[str],
    gzip_norm: float,
    semu_norm: float,
    overlap_len: float,
    window_scores: List[Tuple[int,int,float,list,bool]],
    brand_notes: str | None = None
) -> str:
    # Compact window summary
    ws_lines = []
    for i, (s, e, sc, contrib, _flag) in enumerate(window_scores[:12]):
        tops = ", ".join([f"Q{qi+1}@{qs:.2f}" for (qi, _q, qs) in contrib[:3]]) or "—"
        ws_lines.append(f"W{i+1}[{s}:{e}] score={sc:.2f} tops={tops}")
    ws_block = "\n".join(ws_lines) if ws_lines else "None"

    q_block = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(queries)])
    brand = f"\nBRAND GUARDRAILS:\n{brand_notes}\n" if brand_notes else ""

    prompt = f"""You are an editor for web copy. Rewrite the passage to improve three axes while staying natural:
- Relevance (semantic overlap with queries), 
- Semantic Uniques (breadth),
- Density (avoid fluff), keeping a balanced overall tone.

Constraints:
- Keep it concise, skimmable, and on brand.
- Avoid keyword stuffing; use natural variations.
- Keep meaning accurate; no fabricated facts.
- Use simple language (aim B2 reading level).
- Keep ~10–25% shorter if possible, unless brevity harms relevance.{brand}

QUERIES:
{q_block}

CURRENT METRICS:
- Density (gzip_norm): {gzip_norm:.2f}
- Uniques (semu_norm): {semun_norm:.2f}
- Overlap (overlap): {overlap_len:.2f}

WINDOW SUMMARY (strongest spans):
{ws_block}

PASSAGE (rewrite this):
\"\"\"{passage[:6000]}\"\"\"  # truncated for context safety

Return JSON with keys:
- "reasoning": 2–4 bullets stating what you changed and why.
- "rewrite": the improved passage only.
"""
    return prompt

def call_gpt35(api_key: str, prompt: str, temperature: float = 0.2, model: str = GPT35_DEFAULT) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": "You are a precise, concise web editor."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    try:
        r = requests.post(OPENAI_CHAT_URL, headers=headers, data=json.dumps(payload), timeout=60)
        if r.status_code != 200:
            return {"error": f"OpenAI API error {r.status_code}: {r.text[:300]}"}
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def render_diff(old: str, new: str):
    old_lines = [l.rstrip() for l in old.splitlines()]
    new_lines = [l.rstrip() for l in new.splitlines()]
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="original", tofile="rewrite", lineterm="")
    out = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f"<div style='color:#0b7a0b'> {html.escape(line)}</div>")
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"<div style='color:#b00020'> {html.escape(line)}</div>")
        else:
            out.append(f"<div style='color:#666'> {html.escape(line)}</div>")
    if not out:
        out = ["<div style='color:#666'>(No line-level changes)</div>"]
    st.markdown("<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:0.9rem;'>"
                + "\n".join(out) + "</div>", unsafe_allow_html=True)

# ----------------- UI -----------------
st.set_page_config(page_title="Semantic Overlap & Density (Editor Mode)", layout="wide")
st.title("Semantic Overlap & Density — Editor Mode (FastEmbed)")

with st.expander("📘 How to use", expanded=False):
    st.markdown("""
**Run analysis**, then optionally **get a GPT-3.5 rewrite** (only if you paste an API key).  
No key = metrics + highlights only.
""")

def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

with st.sidebar:
    st.markdown("### 🤖 Configuration")
    model_name = st.selectbox("Embedding Model",
        ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"],
        index=init_state("model_idx", 0))
    st.session_state["model_idx"] = ["BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"].index(model_name)

    st.markdown("### 🪟 Windowing")
    win_size = st.slider("Sentence window size", 1, 6, init_state("win_size", 3))
    st.session_state["win_size"] = win_size
    stride   = st.slider("Window stride", 1, 6, init_state("stride", 2))
    st.session_state["stride"] = stride

    border_thresh = st.slider("Window border threshold", 0.0, 1.0, init_state("border_thresh", float(MIN_WINDOW_BORDER_SCORE)), 0.05)
    st.session_state["border_thresh"] = border_thresh
    MIN_WINDOW_BORDER_SCORE = border_thresh

    st.markdown("### 🎨 Visualization Mode")
    viz_mode = st.radio("Choose visualization focus",
                        ["Windows Only", "Coverage Stripes Only", "Both (cluttered)"],
                        index=init_state("viz_mode_idx", 0))
    st.session_state["viz_mode_idx"] = ["Windows Only", "Coverage Stripes Only", "Both (cluttered)"].index(viz_mode)
    if viz_mode == "Windows Only":
        show_stripes, show_windows = False, True
    elif viz_mode == "Coverage Stripes Only":
        show_stripes, show_windows = True, False
    else:
        show_stripes, show_windows = True, True
    st.session_state["show_stripes"] = show_stripes
    st.session_state["show_windows"] = show_windows

    st.markdown("### 🎨 Additional Overlays")
    show_sentence_chips = st.checkbox("Sentence chips with deltas", value=init_state("show_sentence_chips", True))
    st.session_state["show_sentence_chips"] = show_sentence_chips
    show_filler = st.checkbox("Highlight filler/hedging", value=init_state("show_filler", True))
    st.session_state["show_filler"] = show_filler
    overstuff_limit = st.number_input("Over-stuffing threshold (exact repeats)", 0, 10, init_state("overstuff_limit", 2))
    st.session_state["overstuff_limit"] = overstuff_limit

    st.markdown("### ✂️ Editor helpers")
    if st.button("Trim filler words"): st.session_state["apply_trim_filler"] = True
    if st.button("Collapse repeated spaces"): st.session_state["apply_collapse_spaces"] = True

    st.markdown("### ✨ Optional: GPT-3.5 Rewrite")
    openai_key = st.text_input("OpenAI API Key", type="password", value=init_state("openai_key", ""))
    st.session_state["openai_key"] = openai_key
    gpt_temp = st.slider("Creativity (temperature)", 0.0, 1.0, init_state("gpt_temp", 0.2), 0.05)
    st.session_state["gpt_temp"] = gpt_temp
    brand_notes = st.text_area("Brand guardrails (tone, style, banned words, etc.)",
                               value=init_state("brand_notes", ""))
    st.session_state["brand_notes"] = brand_notes
    gpt_model = st.text_input("GPT-3.5 model", value=init_state("gpt_model", GPT35_DEFAULT),
                              help="Defaults to gpt-3.5-turbo-0125. Change if needed.")
    st.session_state["gpt_model"] = gpt_model

# Inputs
colA, colB = st.columns([1,1])
with colA:
    default_passage = st.session_state.get("passage_text", "")
    passage = st.text_area("Passage", height=260, value=default_passage, placeholder="Paste your content here…")
with colB:
    default_queries = st.session_state.get("queries_text", "")
    raw_queries = st.text_area("Queries (one per line, up to 10)", height=260, value=default_queries,
                               placeholder="e.g.\nluxury resort whistler\nski-in ski-out suites\nspa and wellness")
    queries = [q.strip() for q in raw_queries.splitlines() if q.strip()][:MAX_QUERIES]

# Apply one-click editor helpers (persistent)
if st.session_state.get("apply_trim_filler"):
    passage = re.sub(FILLER_WORDS, "", passage)
    passage = re.sub(r"\s{2,}", " ", passage).strip()
    st.session_state["apply_trim_filler"] = False
if st.session_state.get("apply_collapse_spaces"):
    passage = re.sub(r"\s{2,}", " ", passage)
    st.session_state["apply_collapse_spaces"] = False

st.session_state["passage_text"] = passage
st.session_state["queries_text"] = raw_queries

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
        ov_len, win_scores, sims, q_labels, w_labels, wins = overlap_embed(
            passage, queries, model_name=model_name, win_size=win_size, stride=stride
        )
        final = geometric_mean([gzip_norm, semu_norm, ov_len])

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gzip (density, norm)", f"{gzip_norm:.2f}")
    m2.metric("Semantic Uniques (norm)", f"{semun_norm:.2f}")
    m3.metric("Overlap (relevance)",    f"{ov_len:.2f}")
    m4.metric("Content Balance Score",  f"{final:.2f}")

    if tok_count < 25:
        st.info("Very short passages (< 25 tokens) can be unstable.")

    st.markdown("---")

    # --- Annotated passage ---
    st.subheader("Annotated Passage")
    html_block = render_passage(
        passage=passage,
        window_scores=win_scores,
        wins=wins,
        sims=sims,
        queries=queries,
        show_per_query_stripes=st.session_state.get("show_stripes", True),
        show_filler_flags=st.session_state.get("show_filler", True),
        overstuff_threshold=int(st.session_state.get("overstuff_limit", 2)),
        show_windows=st.session_state.get("show_windows", True)
    )
    st.markdown("<div style='line-height:1.8; font-size:1.05rem;'>"+html_block+"</div>", unsafe_allow_html=True)

    # Legend + Query key
    st.markdown("### Legend")
    legend_cols = st.columns(min(4, max(1, len(win_scores))))
    for i, ws in enumerate(win_scores):
        _, _, score, contrib, _ = ws
        color = WINDOW_PALETTE[i % len(WINDOW_PALETTE)]
        with legend_cols[i % len(legend_cols)]:
            st.markdown(
                f"<div style='display:inline-block;width:14px;height:14px;background:{color};"
                f"border-radius:2px;margin-right:8px;vertical-align:middle;'></div>"
                f"<span><b>W{i+1}</b> (score {score:.2f})</span>",
                unsafe_allow_html=True
            )
    st.markdown("### Query Key")
    st.markdown(", ".join([f"**Q{i+1}**: <span style='color:{QUERY_PALETTE[i%len(QUERY_PALETTE)]}'>{html.escape(q)}</span>"
                           for i, q in enumerate(queries)]) , unsafe_allow_html=True)

    # --- Sentence chips with deltas ---
    if st.session_state.get("show_sentence_chips", True):
        st.markdown("### Sentence Chips (Δ vs previous run)")
        sents = split_sents(passage)
        chips = []
        for (s, e, txt) in sents:
            m = sentence_metrics(txt, queries, model_name=model_name)
            chips.append((txt, m))
        prev = st.session_state.get("prev_sentence_scores", {})
        if not isinstance(prev, dict):
            prev = {}
            st.session_state["prev_sentence_scores"] = prev
        curr = {}
        for txt, m in chips:
            key = hashlib.md5(txt.encode("utf-8")).hexdigest()
            curr[key] = m
            d = prev.get(key, None)
            def fmt(val): return f"{val:.2f}"
            def delta(curr, prev): 
                return f"{(curr - prev):+0.02f}" if prev is not None else "—"
            density = fmt(m["density"]); d_d = delta(m["density"], d["density"]) if d else "—"
            uniq    = fmt(m["uniques"]); d_u = delta(m["uniques"], d["uniques"]) if d else "—"
            ov      = fmt(m["overlap"]); d_o = delta(m["overlap"], d["overlap"]) if d else "—"
            st.markdown(
                f"<div style='margin:4px 0; padding:6px 8px; border:1px solid #eee; border-radius:6px;'>"
                f"<div style='font-size:0.95rem; margin-bottom:4px'>{html.escape(txt)}</div>"
                f"<div style='font-size:0.85rem; color:#444'>"
                f"Overlap <b>{ov}</b> (<span style='color:#666'>{d_o}</span>) • "
                f"Uniques <b>{uniq}</b> (<span style='color:#666'>{d_u}</span>) • "
                f"Density <b>{density}</b> (<span style='color:#666'>{d_d}</span>)"
                f"</div></div>",
                unsafe_allow_html=True
            )
        st.session_state["prev_sentence_scores"] = curr

    # --- Repetition meter + synonym nudges ---
    st.markdown("### Repetition Meter & Synonym Nudges")
    reps = top_ngrams(passage, n=10)
    if reps:
        cols = st.columns(2)
        with cols[0]:
            st.write({k: reps[k] for k in list(reps.keys())[:10]})
        with cols[1]:
            unigram_terms = [k for k in reps.keys() if " " not in k][:5]
            hints = synonym_nudges(unigram_terms)
            if hints:
                for term, alt in hints.items():
                    st.write(f"**{term}** → " + ", ".join(alt))
            else:
                st.write("No synonym nudges found for top terms.")
    else:
        st.write("No repeated terms found.")

    # --- Edit plan (actionable) ---
    st.markdown("### Edit Plan")
    prio = query_priorities(queries, st.session_state.get("query_weights", {}))
    plan = build_edit_plan(
        passage=passage,
        queries=queries,
        sims=sims,
        wins=wins,
        window_scores=win_scores,
        priorities=prio,
        filler_count_limit=int(st.session_state.get("overstuff_limit", 2))
    )
    with st.expander("✅ Suggested next edits", expanded=True):
        render_edit_plan(plan, passage)

    # ---------------- GPT-3.5 Rewrite (only if API key present) ----------------
    if st.session_state.get("openai_key", "").strip():
        st.markdown("## ✨ GPT-3.5 Rewrite (optional)")
        st.caption("Runs only when you click the button. We send the passage, queries, and high-level scores—no analytics beyond that.")
        if st.button("Generate GPT-3.5 Rewrite"):
            with st.spinner("Calling GPT-3.5…"):
                prompt = build_llm_prompt(
                    passage=passage,
                    queries=queries,
                    gzip_norm=gzip_norm,
                    semu_norm=semun_norm,
                    overlap_len=ov_len,
                    window_scores=win_scores,
                    brand_notes=st.session_state.get("brand_notes", "")
                )
                resp = call_gpt35(
                    api_key=st.session_state["openai_key"],
                    prompt=prompt,
                    temperature=float(st.session_state.get("gpt_temp", 0.2)),
                    model=st.session_state.get("gpt_model", GPT35_DEFAULT)
                )
            if "error" in resp:
                st.error(resp["error"])
            else:
                reasoning = resp.get("reasoning", "")
                rewrite = resp.get("rewrite", "")
                st.markdown("### Model’s reasoning (summary)")
                if isinstance(reasoning, list):
                    st.markdown("\n".join([f"- {html.escape(x)}" for x in reasoning]), unsafe_allow_html=True)
                else:
                    st.markdown(textwrap.indent(str(reasoning), "- "), unsafe_allow_html=False)

                st.markdown("### Rewritten passage")
                st.text_area("Rewrite", value=rewrite, height=220)

                st.markdown("### Diff vs original")
                render_diff(passage, rewrite)
    else:
        st.info("🔒 Paste an OpenAI API key in the sidebar to enable the optional GPT-3.5 rewrite.")
