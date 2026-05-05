"""
Snippet Detector — Line-Level Viral Analysis
=============================================
Scores every individual lyric line for TikTok virality and pinpoints
the exact lines that would go viral, not just a window.

Per-line scoring axes (weighted):
  1. Repetition count   (0.28) — how many times this line recurs in the song
  2. Energy density     (0.22) — high-energy words per line
  3. Rhyme match        (0.15) — end-rhymes with adjacent line
  4. Brevity            (0.13) — 4–9 words → optimal for lip-sync / text overlay
  5. Section bonus      (0.12) — [Chorus] / [Hook] lines get a lift
  6. Emotional punch    (0.10) — strong positive OR negative sentiment

Output:
  - viral_line      : single highest-scoring line
  - viral_clip      : 5-line window centred on viral_line (the TikTok clip)
  - annotated_lines : every line with its score + why (for UI heat-map)
  - virality_report : full breakdown for the song
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# ── Word sets ──────────────────────────────────────────────────────────────────

ENERGY_WORDS = {
    "fire", "lit", "hype", "wild", "crazy", "turnt", "vibe", "bounce",
    "drop", "bang", "banger", "loud", "scream", "jump", "move", "go",
    "fast", "hard", "rage", "beast", "king", "queen", "boss", "flex",
    "drip", "swag", "pop", "rock", "hit", "unstoppable", "epic", "run",
    "rise", "shine", "burn", "blaze", "fly", "rush", "push", "break",
}

POSITIVE_WORDS = {
    "love", "happy", "joy", "smile", "beautiful", "wonderful", "amazing",
    "great", "good", "best", "perfect", "shine", "bright", "warm", "sweet",
    "dream", "hope", "win", "glory", "bless", "grace", "light", "gold",
    "free", "peace", "together", "forever", "always", "paradise", "heaven",
}

NEGATIVE_WORDS = {
    "hate", "pain", "cry", "tears", "hurt", "break", "lost", "alone",
    "dark", "cold", "dead", "fall", "fail", "lie", "cheat", "betray",
    "gone", "miss", "sad", "empty", "numb", "sorrow", "grief", "regret",
    "fear", "wrong", "bad", "worst", "hell", "die", "kill", "blood",
}

_SECTION_RE   = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)
HOOK_SECTIONS = {"chorus", "hook", "refrain", "bridge", "pre-chorus"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _last_word_ending(line: str, n: int = 3) -> str:
    words = line.strip().split()
    return words[-1].lower()[-n:] if words else ""


def _rhymes(a: str, b: str) -> bool:
    ea, eb = _last_word_ending(a), _last_word_ending(b)
    return len(ea) >= 2 and ea == eb


def _energy(line: str) -> float:
    words = set(line.lower().split())
    return min(len(words & ENERGY_WORDS) / 3.0, 1.0)


def _sentiment_strength(line: str) -> float:
    """Abs(positive - negative) intensity — emotional punch regardless of polarity."""
    words = set(line.lower().split())
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
    return min(abs(pos - neg) / 3.0, 1.0)


def _brevity(line: str) -> float:
    n = len(line.split())
    if 4 <= n <= 9:
        return 1.0
    if n < 4:
        return 0.5 + n * 0.1
    return max(0.0, 1.0 - (n - 9) * 0.08)


# ── Annotated line model ───────────────────────────────────────────────────────

def _parse_annotated(lyrics: str) -> list[dict]:
    """
    Return list of dicts, one per non-blank line:
      idx, section, text, is_header
    """
    raw_lines      = lyrics.split("\n")
    current_sec    = "intro"
    annotated      = []
    lyric_idx      = 0

    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        m = _SECTION_RE.match(s)
        if m:
            current_sec = m.group(1).strip().lower()
            annotated.append({
                "idx": None, "section": current_sec,
                "text": s, "is_header": True,
            })
        else:
            annotated.append({
                "idx": lyric_idx, "section": current_sec,
                "text": s, "is_header": False,
            })
            lyric_idx += 1

    return annotated


# ── Per-line scoring ───────────────────────────────────────────────────────────

WEIGHTS = {
    "repetition":       0.28,
    "energy":           0.22,
    "rhyme":            0.15,
    "brevity":          0.13,
    "section_bonus":    0.12,
    "emotional_punch":  0.10,
}


def score_all_lines(lyrics: str) -> list[dict]:
    """
    Score every lyric line (non-header) in the song.

    Returns list of dicts:
      idx, section, text, score, factors, label, why
    where `why` is a short human-readable string for the UI.
    """
    annotated   = _parse_annotated(lyrics)
    lyric_lines = [a for a in annotated if not a["is_header"]]

    if not lyric_lines:
        return []

    # Global repetition counts
    line_counts = Counter(a["text"].strip().lower() for a in lyric_lines)
    max_count   = max(line_counts.values(), default=1)

    scored = []
    texts  = [a["text"] for a in lyric_lines]  # for neighbour rhyme lookup

    for i, ann in enumerate(lyric_lines):
        text    = ann["text"]
        sec     = ann["section"]
        key     = text.strip().lower()

        # 1. Repetition
        rep_raw   = line_counts[key]
        rep_score = (rep_raw - 1) / max(max_count - 1, 1) if max_count > 1 else 0.0

        # 2. Energy
        energy_sc = _energy(text)

        # 3. Rhyme (with prev or next lyric line)
        prev_text = texts[i - 1] if i > 0          else ""
        next_text = texts[i + 1] if i < len(texts) - 1 else ""
        rhyme_sc  = 1.0 if (_rhymes(text, prev_text) or _rhymes(text, next_text)) else 0.0

        # 4. Brevity
        brev_sc = _brevity(text)

        # 5. Section bonus
        sec_sc = 1.0 if any(k in sec for k in HOOK_SECTIONS) else 0.0

        # 6. Emotional punch
        emo_sc = _sentiment_strength(text)

        factors = {
            "repetition":      round(rep_score, 3),
            "energy":          round(energy_sc, 3),
            "rhyme":           round(rhyme_sc,  3),
            "brevity":         round(brev_sc,   3),
            "section_bonus":   round(sec_sc,    3),
            "emotional_punch": round(emo_sc,    3),
        }

        score = sum(WEIGHTS[k] * factors[k] for k in WEIGHTS)

        # Human label
        if score >= 0.65:
            label = "🔥 Viral"
        elif score >= 0.45:
            label = "⚡ Strong Hook"
        elif score >= 0.28:
            label = "✓ Solid"
        else:
            label = "–"

        # Short why string
        why_parts = []
        if rep_score >= 0.5:
            why_parts.append(f"repeats {rep_raw}×")
        if energy_sc >= 0.33:
            why_parts.append("high energy")
        if rhyme_sc == 1.0:
            why_parts.append("rhymes")
        if sec_sc == 1.0:
            why_parts.append("in chorus/hook")
        if emo_sc >= 0.33:
            why_parts.append("emotional punch")
        if brev_sc >= 0.9:
            why_parts.append("perfect length")

        scored.append({
            "idx":     ann["idx"],
            "section": sec,
            "text":    text,
            "score":   round(score, 4),
            "factors": factors,
            "label":   label,
            "why":     ", ".join(why_parts) if why_parts else "low virality",
            "repeats": rep_raw,
        })

    return scored


# ── Viral moment finder ────────────────────────────────────────────────────────

def find_viral_moment(lyrics: str, clip_lines: int = 5) -> dict:
    """
    Identify:
      - viral_line     : the single highest-scoring line
      - viral_clip     : clip_lines centred on viral_line (the TikTok clip)
      - runner_up      : second-best line
      - all_scored     : every line with scores (for heat-map)
      - reasoning      : bullet-point explanation
      - section        : which section the viral line lives in
    """
    scored = score_all_lines(lyrics)
    if not scored:
        return _empty_viral()

    sorted_lines = sorted(scored, key=lambda x: x["score"], reverse=True)
    viral        = sorted_lines[0]
    runner_up    = sorted_lines[1] if len(sorted_lines) > 1 else None

    # Build clip: clip_lines lines centred on viral line's position
    viral_pos  = next((i for i, s in enumerate(scored) if s["idx"] == viral["idx"]), 0)
    half       = clip_lines // 2
    start      = max(0, viral_pos - half)
    end        = min(len(scored), start + clip_lines)
    start      = max(0, end - clip_lines)   # re-anchor if near end
    clip       = scored[start:end]

    reasoning  = _build_viral_reasoning(viral, runner_up, clip)

    return {
        "viral_line":  viral,
        "runner_up":   runner_up,
        "viral_clip":  clip,
        "all_scored":  scored,
        "reasoning":   reasoning,
        "section":     viral["section"].title(),
    }


def _empty_viral() -> dict:
    return {
        "viral_line":  None,
        "runner_up":   None,
        "viral_clip":  [],
        "all_scored":  [],
        "reasoning":   "No lyric lines found.",
        "section":     "N/A",
    }


def _build_viral_reasoning(viral: dict, runner_up: Optional[dict], clip: list[dict]) -> str:
    lines = []
    sc    = round(viral["score"] * 100)
    lines.append(f"**The TikTok moment lives in the [{viral['section'].title()}]** "
                 f"(viral score: {sc}%)")
    lines.append("")
    lines.append(f"> *\"{viral['text']}\"*")
    lines.append("")

    why_detail = []
    f = viral["factors"]
    if f["repetition"] >= 0.5:
        why_detail.append(f"- **Repeats {viral['repeats']}× in the song** — the brain locks onto lines it hears multiple times.")
    if f["energy"] >= 0.33:
        why_detail.append("- **High-energy language** — words that create excitement or urgency in the listener.")
    if f["rhyme"] == 1.0:
        why_detail.append("- **End-rhyme with adjacent line** — natural flow makes it satisfying to lip-sync.")
    if f["section_bonus"] == 1.0:
        why_detail.append("- **Lives in the Chorus/Hook** — the most re-played section of any song.")
    if f["emotional_punch"] >= 0.33:
        why_detail.append("- **Emotional hit** — strong feeling (love, pain, hype) triggers reaction videos.")
    if f["brevity"] >= 0.9:
        why_detail.append(f"- **Perfect length** ({len(viral['text'].split())} words) — short enough for text overlays and duets.")

    lines.extend(why_detail if why_detail else ["- This line scores highest across all virality factors."])

    if runner_up:
        lines.append(f"\n**Also worth clipping:** *\"{runner_up['text']}\"* "
                     f"({round(runner_up['score']*100)}%) in [{runner_up['section'].title()}]")

    lines.append("\n**TikTok clip:** Use the 5 lines shown in the clip box. "
                 "Start at the beginning of that section for maximum impact.")
    return "\n".join(lines)


# ── Full virality analysis (for pipeline / Viral Blueprint tab) ───────────────

VIRALITY_FACTORS = {
    "hook_strength":   ("Hook Strength",    "Most repeated line frequency",        0.22),
    "energy_score":    ("Energy Score",     "High-energy word density",            0.20),
    "rhyme_density":   ("Rhyme Density",    "End-rhyme pairs across the song",     0.18),
    "repetition_rate": ("Repetition Rate",  "Fraction of lines that repeat",       0.15),
    "chorus_density":  ("Chorus Density",   "Proportion of sections that are hook",0.12),
    "brevity":         ("Line Brevity",     "Average line shortness (< 9 words)",  0.13),
}


def virality_analysis(lyrics: str, lyric_feats: Optional[dict] = None) -> dict:
    from similarity_engine import extract_lyric_features

    feats   = lyric_feats or extract_lyric_features(lyrics)
    vm      = find_viral_moment(lyrics)

    # Brevity from scored lines average
    scored  = vm["all_scored"]
    avg_brev = (sum(s["factors"]["brevity"] for s in scored) / len(scored)) if scored else 0.5
    feats["brevity"] = avg_brev

    factor_scores: dict[str, float] = {k: round(feats.get(k, 0.0), 4)
                                        for k in VIRALITY_FACTORS}

    overall = round(sum(feats.get(k, 0) * w
                        for k, (_, _, w) in VIRALITY_FACTORS.items()), 4)

    top_f = sorted(factor_scores, key=factor_scores.get, reverse=True)[:3]
    weak_f = sorted(factor_scores, key=factor_scores.get)[:2]

    why = ["**Why this song could trend:**"]
    for k in top_f:
        lbl, desc, _ = VIRALITY_FACTORS[k]
        why.append(f"- **{lbl}** ({round(factor_scores[k]*100)}%) — {desc}")
    why.append("\n**Potential weak spots:**")
    for k in weak_f:
        lbl, desc, _ = VIRALITY_FACTORS[k]
        why.append(f"- **{lbl}** ({round(factor_scores[k]*100)}%) — {desc}")

    return {
        "overall_virality":  overall,
        "factor_scores":     factor_scores,
        "virality_label":    _virality_label(overall),
        "viral_moment":      vm,
        "summary":           "\n".join(why),
    }


def _virality_label(score: float) -> str:
    if score >= 0.55:   return "High Viral Potential"
    if score >= 0.35:   return "Moderate Viral Potential"
    return "Low Viral Potential"
