def render_highlighted(passage: str, window_scores: List[Tuple]) -> str:
    """Render passage with window highlighting and unique word coloring"""
    # Get unique words for green highlighting
    unique_words = get_unique_words(passage)
    
    # First apply unique word coloring to the original passage
    def color_unique_words(match):
        word = match.group(0)
        word_lower = word.lower()
        if word_lower in unique_words:
            return f"<span style='color: green; font-weight: bold;'>{word}</span>"
        return word
    
    # Apply unique word coloring to the original passage
    result = re.sub(r'\b\w+\b', color_unique_words, passage)
    
    # Now apply window span highlighting with red dotted borders
    # Sort by start position and apply from end to beginning to avoid position shifts
    sorted_windows = sorted(window_scores, key=lambda x: x[0])
    
    for start, end, score, contributing_queries in reversed(sorted_windows):
        if score > 0.1:
            window_text = result[start:end]
            
            # Format query breakdown
            breakdown_text = " | ".join([f"Q{q_idx+1}:{q_score:.2f}" for q_idx, _, q_score in contributing_queries[:2]])
            annotation = f"<sup style='font-size:0.7em; color:#666;'>({score:.2f} | {breakdown_text})</sup>"
            
            # Use span with inline-block to avoid breaking text flow
            window_html = f"<span style='{color_for_score(score)}; display: inline-block;'>{window_text}{annotation}</span>"
            result = result[:start] + window_html + result[end:]
    
    return result
