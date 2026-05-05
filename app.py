"""
TikTok Viral Detector
Tab 1 — Viral Detector  : paste lyrics → song-level virality score + exact viral line
Tab 2 — Tatiana's Songs : batch analysis from Tatiana_full_lyrics.csv (no API calls)
"""

from __future__ import annotations

import re
from itertools import groupby
from pathlib import Path
import streamlit as st
import pandas as pd

from config import TIKTOK_TRENDING_CSV

TATIANA_CSV = Path(__file__).parent / "Tatiana_full_lyrics.csv"

st.set_page_config(
    page_title="TikTok Viral Detector",
    page_icon="🔥",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def prob_bar(prob: float, width: int = 100) -> str:
    pct   = int(prob * 100)
    color = ("#ff6b6b" if pct >= 75 else
             "#ffd93d" if pct >= 55 else
             "#6bcb77" if pct >= 35 else "#444")
    return (
        f'<div style="display:inline-flex;align-items:center;gap:6px">'
        f'<div style="background:#1a1a1a;border-radius:4px;height:7px;width:{width}px">'
        f'<div style="background:{color};height:7px;border-radius:4px;width:{min(pct,100)}%">'
        f'</div></div>'
        f'<span style="font-size:0.78em;color:{color};font-weight:600">{pct}%</span>'
        f'</div>'
    )

def line_col(prob: float) -> str:
    if prob >= 0.75: return "#ff6b6b"
    if prob >= 0.55: return "#ffd93d"
    if prob >= 0.35: return "#6bcb77"
    return "#3a3a3a"

def viral_badge(prob: float) -> str:
    if prob >= 0.75: return "🔥 Viral"
    if prob >= 0.55: return "⚡ Strong Hook"
    if prob >= 0.35: return "✓ Solid"
    return "–"

def song_badge_color(prob: float) -> str:
    if prob >= 0.65: return "#ff6b6b"
    if prob >= 0.45: return "#ffd93d"
    return "#888"

def _clean_title(title: str) -> str:
    t = str(title).strip().strip("'\"")
    t = re.sub(
        r"\s+('?)(?:Official\s+)?(?:Audio|Video|Lyric\s+Video|Music\s+Video)('?)",
        "", t, flags=re.I,
    ).strip().strip("'\"").strip()
    return t

def models_ready() -> bool:
    return (
        Path("models/viral_line_model.pkl").exists() and
        Path("models/song_viral_model.pkl").exists()
    )

@st.cache_resource(show_spinner="Loading models …")
def get_detector():
    from nlp_models import ViralLineDetector
    return ViralLineDetector()

@st.cache_resource(show_spinner="Loading song virality model …")
def get_song_predictor():
    from nlp_models import SongViralityPredictor
    return SongViralityPredictor()

def render_heatmap(all_lines: list[dict], viral_idx: int) -> None:
    """Section-grouped heat-map — shared by both tabs."""
    st.caption("🔥 ≥75%  ·  ⚡ ≥55%  ·  ✓ ≥35%  ·  – below threshold")
    for sec_name, sec_iter in groupby(all_lines, key=lambda x: x["section"]):
        sec_lines = list(sec_iter)
        peak_prob = max(s["viral_prob"] for s in sec_lines)
        sec_color = ("#ff6b6b" if peak_prob >= 0.75 else
                     "#ffd93d" if peak_prob >= 0.55 else
                     "#6bcb77" if peak_prob >= 0.35 else "#555")
        with st.container(border=True):
            st.markdown(
                f'<div style="color:{sec_color};font-size:0.78em;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.8px;padding-bottom:6px">'
                f'[{sec_name.title()}]'
                f'<span style="color:#666;font-weight:400;margin-left:8px">'
                f'peak {round(peak_prob * 100)}%</span></div>',
                unsafe_allow_html=True,
            )
            for s in sec_lines:
                is_vl  = (s["idx"] == viral_idx)
                col    = line_col(s["viral_prob"])
                weight = "700" if is_vl else "400"
                st.markdown(
                    f'<div style="border-left:3px solid {col};padding-left:8px;margin:2px 0">'
                    f'<span style="color:{col};font-weight:{weight}">'
                    f'{"🔥 " if is_vl else ""}{s["text"]}</span><br>'
                    f'{prob_bar(s["viral_prob"])} '
                    f'<span style="color:#666;font-size:0.73em">{s["label"]} — {s["why"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

def render_song_virality_banner(song_result: dict) -> None:
    """Big banner showing song-level TikTok virality score."""
    prob  = song_result["viral_prob"]
    label = song_result["label"]
    pct   = round(prob * 100)
    color = song_badge_color(prob)

    st.markdown(
        f'<div style="background:#111;border:2px solid {color};border-radius:8px;'
        f'padding:16px 20px;margin-bottom:12px">'
        f'<div style="display:flex;align-items:center;gap:16px">'
        f'<div style="font-size:2em;font-weight:900;color:{color}">{pct}%</div>'
        f'<div>'
        f'<div style="font-size:1.1em;font-weight:700;color:{color}">{label}</div>'
        f'<div style="color:#888;font-size:0.82em">Overall TikTok Viral Score</div>'
        f'</div></div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why this score?", expanded=False):
        st.markdown(song_result["reasoning"])


# ── TikTok video idea engine ───────────────────────────────────────────────────

def generate_tiktok_ideas(viral_line: dict, feat: dict, section: str) -> list[dict]:
    ideas     = []
    line_text = viral_line["text"]
    repeats   = viral_line["repeats"]
    energy    = feat.get("energy_score", 0)
    pos       = feat.get("positive_sent", 0)
    neg       = feat.get("negative_sent", 0)
    hook_str  = feat.get("hook_strength", 0)
    sec_lower = section.lower()

    if repeats >= 2:
        ideas.append({
            "format": "🎤 Lip Sync Hook",
            "idea":   f'Mouth *"{line_text}"* perfectly on beat. Cut to your reaction right after.',
            "timing": "Start filming exactly when the section begins.",
            "effect": "Green screen or simple background.",
            "hashtags": ["#lipsync", "#fyp", "#viral", "#trending"],
        })
    if energy >= 0.3 or "chorus" in sec_lower or "hook" in sec_lower:
        ideas.append({
            "format": "💃 Dance / Movement Challenge",
            "idea":   f'Create a 4-8 count move that hits on *"{line_text}"*.',
            "timing": "Build-up for 2 counts before the line drops, explode on the beat.",
            "effect": "Trending transition or slow-mo at the peak move.",
            "hashtags": ["#dancechallenge", "#choreography", "#fyp", "#duet"],
        })
    if neg >= 0.2 or "verse" in sec_lower or "bridge" in sec_lower:
        ideas.append({
            "format": "🎬 POV / Storytelling",
            "idea":   f'Set a scene around *"{line_text}"*. Show the emotion visually.',
            "timing": "Let the clip run through the full 5-line window.",
            "effect": "Slow motion, black-and-white filter, or cinematic crop.",
            "hashtags": ["#pov", "#storytelling", "#fyp", "#relatable"],
        })
    if repeats >= 3 or hook_str >= 0.15:
        ideas.append({
            "format": "🤝 Duet Invitation",
            "idea":   f'Post *"{line_text}"* and caption it "duet this with your reaction".',
            "timing": "Your video: just the hook, looped or repeated twice.",
            "effect": "React emoji or countdown overlay to prompt duets.",
            "hashtags": ["#duet", "#fyp", "#reaction", "#challenge"],
        })
    if pos >= 0.2 or energy >= 0.4:
        ideas.append({
            "format": "✨ Transformation / Glow Up",
            "idea":   f'Before → After cut timed to *"{line_text}"*.',
            "timing": "Before state in silence, drop the audio exactly on this line.",
            "effect": "Flash cut, light leak, or outfit change transition.",
            "hashtags": ["#glowup", "#transformation", "#fyp", "#trending"],
        })
    words = len(line_text.split())
    if words <= 9:
        ideas.append({
            "format": "📝 Text Overlay / Relatable Moment",
            "idea":   f'Film a mundane moment and overlay *"{line_text}"* as the caption.',
            "timing": "Casual B-roll. Let the line land as a punchline at the end.",
            "effect": "Bold text, jump cut, or freeze frame on the last word.",
            "hashtags": ["#relatable", "#fyp", "#humor", "#vibe"],
        })
    return ideas[:5]


# ══════════════════════════════════════════════════════════════════════════════
# Tatiana analysis (cached — no API calls, pure NLP)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _analyse_tatiana() -> list[dict]:
    """Load Tatiana_full_lyrics.csv and run both models on every song."""
    df = pd.read_csv(TATIANA_CSV, index_col=0)
    df = df[df.index != "song"]  # drop header row if present

    detector  = get_detector()
    predictor = get_song_predictor()

    results = []
    for song_raw, row in df.iterrows():
        song   = str(song_raw).strip()
        lyrics = str(row["full_tatiana_demaria_lyrics"]).strip()
        if not lyrics or lyrics.lower() == "nan" or lyrics.lower() == "lyrics":
            continue

        line_res = detector.predict(lyrics)
        song_res = predictor.predict(lyrics, line_res)

        results.append({
            "song":        song,
            "lyrics":      lyrics,
            "viral_prob":  song_res["viral_prob"],
            "label":       song_res["label"],
            "reasoning":   song_res["reasoning"],
            "viral_line":  line_res["viral_line"],
            "viral_clip":  line_res["viral_clip"],
            "all_lines":   line_res["all_lines"],
            "runner_up":   line_res["runner_up"],
        })

    results.sort(key=lambda x: x["viral_prob"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Page header + tabs
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔥 TikTok Viral Detector")
tab1, tab2 = st.tabs(["🔍 Viral Detector", "🎤 Tatiana DeMaria"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Paste any lyrics
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(
        "Paste any song's lyrics. The model first scores **the whole song** for TikTok "
        "viral potential, then finds the **exact line** most likely to blow up, matches "
        "trending TikTok songs, and gives **video ideas** ready to film."
    )
    st.divider()

    ic1, ic2, _ = st.columns([1, 1, 2])
    with ic1:
        artist = st.text_input("Artist (optional)", placeholder="e.g. Sabrina Carpenter")
    with ic2:
        song = st.text_input("Song title (optional)", placeholder="e.g. Espresso")

    lyrics = st.text_area(
        "Paste full lyrics here ✱",
        height=260,
        placeholder=(
            "[Verse 1]\nI can't keep running from what I started\n"
            "[Chorus]\nThat's that me espresso\n"
            "I'm working late 'cause I'm a singer\n"
        ),
    )

    run = st.button("🔍 Analyse Lyrics", type="primary")

    if run:
        if not lyrics.strip():
            st.error("Paste some lyrics first.")
            st.stop()
        if not models_ready():
            st.error("Models not trained. Run `python train.py` in the terminal first.")
            st.stop()

        with st.spinner("Scoring …"):
            detector  = get_detector()
            predictor = get_song_predictor()
            line_res  = detector.predict(lyrics)
            song_res  = predictor.predict(lyrics, line_res)

        vline = line_res["viral_line"]
        if not vline:
            st.warning("Could not parse lyrics — ensure the input has actual lyric lines.")
            st.stop()

        from similarity_engine import extract_lyric_features
        feats = extract_lyric_features(lyrics)

        # ── Song-level virality banner ────────────────────────────────────────
        st.markdown("### Will this song go viral on TikTok?")
        render_song_virality_banner(song_res)
        st.divider()

        # ── ROW 1: Viral line + clip ──────────────────────────────────────────
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### 🔥 The Viral Line")
            with st.container(border=True):
                st.markdown(
                    f"<p style='font-size:1.3em;font-weight:700;color:#ff6b6b;line-height:1.4'>"
                    f"&ldquo;{vline['text']}&rdquo;</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(prob_bar(vline["viral_prob"], width=200), unsafe_allow_html=True)
                st.caption(
                    f"Section: [{vline['section'].title()}]  ·  "
                    f"Repeats **{vline['repeats']}×** in song  ·  {vline['why']}"
                )
            st.markdown(line_res["reasoning"])
            if line_res["runner_up"]:
                ru = line_res["runner_up"]
                st.markdown(
                    f"**Runner-up** ({round(ru['viral_prob']*100)}%): "
                    f"*\"{ru['text']}\"* — [{ru['section'].title()}]"
                )

        with col2:
            st.markdown("### 🎬 Your TikTok Clip")
            st.caption("Film exactly this 5-line section. 🔥 = viral line.")
            clip_lines = []
            for s in line_res["viral_clip"]:
                marker = "🔥 " if s["idx"] == vline["idx"] else "   "
                clip_lines.append(f"{marker}{s['text']}")
            st.code("\n".join(clip_lines), language=None)
            sa, sb, sc_ = st.columns(3)
            sa.metric("Viral Prob",   f"{round(vline['viral_prob']*100)}%")
            sb.metric("Hook Repeats", f"{vline['repeats']}×")
            sc_.metric("Section",     vline["section"].title())

        st.divider()

        # ── ROW 2: Full heat-map ──────────────────────────────────────────────
        with st.expander("📊 Every Line Scored — Full Heat-Map", expanded=False):
            render_heatmap(line_res["all_lines"], vline["idx"])

        st.divider()

        # ── ROW 3: Matching TikTok songs ──────────────────────────────────────
        st.markdown("### 🎵 Matching TikTok Songs")
        st.caption("Trending songs that share lyrical DNA with yours.")

        with st.spinner("Searching TikTok …"):
            try:
                from trending_finder import find_trending_similar
                matches = find_trending_similar(
                    lyrics, query_artist=artist, query_song=song, top_k=5,
                )
            except Exception as e:
                matches = []
                st.caption(f"TikTok search error: {e}")

        if matches:
            kw = matches[0].get("keywords_used", "")
            if kw:
                st.caption(f"Lyrics keywords used to search TikTok: `{kw}`")
            for i, m in enumerate(matches, 1):
                plays       = int(m.get("play_count", 0))
                likes       = int(m.get("like_count", 0))
                why_match   = m.get("why_match", "")
                matched_kws = m.get("matched_keywords", [])
                with st.container(border=True):
                    mc1, mc2, mc3 = st.columns([4, 1, 1])
                    mc1.markdown(f"**{i}. {m['song']}** — {m['artist']}")
                    mc2.metric("Plays", f"{plays:,}")
                    mc3.metric("Likes", f"{likes:,}")
                    if why_match:
                        st.caption(f"Why it matches: {why_match}")
                    if matched_kws:
                        tags_html = " ".join(
                            f'<span style="background:#1a1a2e;color:#4d96ff;padding:2px 7px;'
                            f'border-radius:10px;font-size:0.75em">{t}</span>'
                            for t in matched_kws[:5]
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.info("No TikTok matches returned — quota may be limited or API key missing.")

        st.divider()

        # ── ROW 4: Video ideas ────────────────────────────────────────────────
        st.markdown("### 💡 TikTok Video Ideas")
        ideas = generate_tiktok_ideas(vline, feats, vline["section"])
        for idea in ideas:
            with st.container(border=True):
                st.markdown(f"#### {idea['format']}")
                st.markdown(f"**Concept:** {idea['idea']}")
                ta, tb = st.columns(2)
                with ta:
                    st.markdown(f"**Timing:** {idea['timing']}")
                with tb:
                    st.markdown(f"**Effect / Edit:** {idea['effect']}")
                st.markdown(
                    "**Hashtags:** " + " ".join(
                        f'<span style="background:#1a1a2e;color:#4d96ff;padding:2px 8px;'
                        f'border-radius:10px;font-size:0.8em">{h}</span>'
                        for h in idea["hashtags"]
                    ),
                    unsafe_allow_html=True,
                )

        st.divider()
        with st.expander("⚙️ Pipeline & Model Controls", expanded=False):
            st.caption("Rebuild catalog or retrain models if needed.")
            p1, p2, p3 = st.columns(3)
            with p1:
                if st.button("Retrain Models", use_container_width=True):
                    import subprocess, sys as _sys
                    with st.spinner("Training …"):
                        r = subprocess.run(
                            [_sys.executable, "train.py"],
                            capture_output=True, text=True,
                        )
                    if r.returncode == 0:
                        st.success("Retrained.")
                        st.cache_resource.clear()
                    else:
                        st.error(r.stderr[-400:])
            with p2:
                if st.button("Refresh TikTok Trending", use_container_width=True):
                    with st.spinner("Fetching …"):
                        from tiktok_fetcher import get_trending_songs
                        df_tt = get_trending_songs(count=300, fetch_lyrics=True)
                    st.success(f"{len(df_tt)} songs fetched." if not df_tt.empty else "No results.")
            with p3:
                if st.button("Full Pipeline + Retrain", use_container_width=True):
                    with st.spinner("Running …"):
                        from pipeline import build_catalog
                        import subprocess, sys as _sys
                        build_catalog()
                        subprocess.run([_sys.executable, "train.py"], check=True)
                    st.success("Done.")
                    st.cache_resource.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tatiana DeMaria
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        "Viral analysis for **Tatiana DeMaria's** full catalog using the trained NLP models. "
        "Each song gets a **TikTok Viral Score** (will it go viral?) and the "
        "**exact lyric line** most likely to blow up."
    )
    st.divider()

    if not models_ready():
        st.error("Models not trained. Run `python train.py` in the terminal first.")
        st.stop()

    if not TATIANA_CSV.exists():
        st.error(f"Lyrics file not found: `{TATIANA_CSV}`")
        st.stop()

    with st.spinner("Scoring Tatiana's catalog …"):
        songs = _analyse_tatiana()

    if not songs:
        st.warning("No songs found in the lyrics CSV.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    probs     = [s["viral_prob"] for s in songs]
    avg_prob  = sum(probs) / len(probs)
    high_ct   = sum(1 for p in probs if p >= 0.65)
    mid_ct    = sum(1 for p in probs if 0.45 <= p < 0.65)
    low_ct    = sum(1 for p in probs if p < 0.45)
    top_song  = songs[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Songs Analysed",   str(len(songs)))
    m2.metric("Avg Viral Score",   f"{round(avg_prob * 100)}%")
    m3.metric("🔥 High Potential", str(high_ct))
    m4.metric("⚡ Moderate",       str(mid_ct))

    st.divider()

    # ── Top pick callout ──────────────────────────────────────────────────────
    st.markdown("### Most Likely to Go Viral")
    with st.container(border=True):
        ca, cb = st.columns([3, 2])
        with ca:
            color = song_badge_color(top_song["viral_prob"])
            st.markdown(
                f'<div style="font-size:1.4em;font-weight:800;color:{color}">'
                f'{top_song["song"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:1.1em;font-weight:700;color:{color}">'
                f'{top_song["label"]}  —  {round(top_song["viral_prob"]*100)}%</div>',
                unsafe_allow_html=True,
            )
        with cb:
            if top_song["viral_line"]:
                vl = top_song["viral_line"]
                st.markdown(
                    f'<div style="border-left:3px solid #ff6b6b;padding-left:10px;'
                    f'font-style:italic;color:#ff6b6b;font-size:0.95em">'
                    f'&ldquo;{vl["text"]}&rdquo;</div>'
                    f'<div style="color:#888;font-size:0.78em;margin-top:4px">'
                    f'Line viral prob: {round(vl["viral_prob"]*100)}%  ·  '
                    f'[{vl["section"].title()}]</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Ranked cards for all songs ────────────────────────────────────────────
    st.markdown("### All Songs — Ranked by TikTok Viral Score")
    st.caption("Expand any song for the full line-by-line heat-map and TikTok clip.")

    for entry in songs:
        prob   = entry["viral_prob"]
        color  = song_badge_color(prob)
        vl     = entry["viral_line"]
        label  = entry["label"]

        expander_label = (
            f"{entry['song']}  —  {round(prob*100)}%  {label}"
        )

        with st.expander(expander_label, expanded=False):

            # Song virality banner
            render_song_virality_banner({
                "viral_prob": prob,
                "label":      label,
                "reasoning":  entry["reasoning"],
            })

            if not vl:
                st.info("No scoreable lyric lines found.")
                continue

            # Viral line hero
            ec1, ec2 = st.columns([1, 1], gap="large")
            with ec1:
                st.markdown("**🔥 Viral Line to Film**")
                with st.container(border=True):
                    st.markdown(
                        f'<div style="font-size:1.15em;font-weight:700;color:#ff6b6b">'
                        f'&ldquo;{vl["text"]}&rdquo;</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(prob_bar(vl["viral_prob"], width=180), unsafe_allow_html=True)
                    st.caption(
                        f'[{vl["section"].title()}]  ·  Repeats **{vl["repeats"]}×**  ·  {vl["why"]}'
                    )
                if entry["runner_up"]:
                    ru = entry["runner_up"]
                    st.markdown(
                        f"**Runner-up** ({round(ru['viral_prob']*100)}%): "
                        f"*\"{ru['text']}\"*"
                    )

            with ec2:
                st.markdown("**🎬 TikTok Clip (5 lines)**")
                clip_lines = []
                for s in entry["viral_clip"]:
                    marker = "🔥 " if s["idx"] == vl["idx"] else "   "
                    clip_lines.append(f"{marker}{s['text']}")
                st.code("\n".join(clip_lines), language=None)
                sa, sb, sc_ = st.columns(3)
                sa.metric("Viral Prob",   f"{round(vl['viral_prob']*100)}%")
                sb.metric("Repeats",      f"{vl['repeats']}×")
                sc_.metric("Section",     vl["section"].title())

            # Full heat-map
            st.markdown("**📊 Full Heat-Map — Every Line Scored**")
            render_heatmap(entry["all_lines"], vl["idx"])
