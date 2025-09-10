# app.py
import re
import html
import os
import json
import random
from typing import List, Optional

import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
CSV_PATH = "flashcards_mcq_cleaned_all_v4.csv"  # <— point at your MCQ CSV
PROGRESS_PATH = "progress_flashcards.json"

QUESTION_COL_CANDIDATES = [
    "question",
    "Question",
    "Q",
    "prompt",
    "front",
    "text",
    "stem",
]
ANSWER_COL_CANDIDATES = [
    "answer",
    "Answer",
    "A",
    "solution",
    "Solution",
    "back",
]  # 'back' included
CHOICES_COL_CANDIDATES = ["choices", "Choices", "options", "Options", "mcq", "MCQ"]

st.set_page_config(page_title="Flashcards", layout="wide")

# Back-compat for older Streamlit
if not hasattr(st, "rerun"):
    st.rerun = st.experimental_rerun  # type: ignore[attr-defined]

# ---------- STYLES ----------
st.markdown(
    """
<style>
.topbar {
  padding: .5rem 0 .25rem 0;
  border-bottom: 1px solid rgba(255,255,255,.1);
}
.topbar .stButton>button { height: 40px; font-weight: 600; width: 100%; }
.search-box input { height: 40px; }
.question-card {
  padding: 1rem; border: 1px solid rgba(255,255,255,.1);
  border-radius: 12px; background: rgba(255,255,255,.02);
}
.answer {
  margin-top: .75rem; padding: .75rem 1rem;
  border-left: 3px solid rgba(200,200,200,.5);
  background: rgba(255,255,255,.04); border-radius: 8px;
}
.feedback {
  margin-top: .5rem; padding: .5rem .75rem;
  border-left: 3px solid rgba(255,255,255,.3);
  background: rgba(255,255,255,.03); border-radius: 8px;
}
.badge { padding: .1rem .4rem; border-radius: 6px; background: rgba(255,255,255,.08); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- HELPERS ----------
def pick_col(df, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if len(df.columns) >= 2:
        return df.columns[0] if candidates is QUESTION_COL_CANDIDATES else df.columns[1]
    raise ValueError("Could not find expected columns in CSV.")


def md_linebreaks(s: str) -> str:
    parts = str(s).split("\n\n")
    parts = [p.replace("\n", "  \n") for p in parts]
    return "\n\n".join(parts)


def extract_qnum_label(qtext: str, fallback_idx: int) -> str:
    s = str(qtext).strip()
    m = re.search(r"\bQ[^0-9]{0,3}(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return f"Q{int(m.group(1))}"
    m2 = re.match(r"^\s*(\d+)[\).: ]", s)
    if m2:
        return f"Q{int(m2.group(1))}"
    return f"Q{fallback_idx+1}"


def parse_target_index(user_input: str, ids: List[str]) -> Optional[int]:
    """Accepts '47', 'Q47', 'q 47', etc. Returns zero-based index or None."""
    if not user_input:
        return None
    s = user_input.strip().upper().replace(" ", "")
    m = re.match(r"^(?:Q)?(\d+)$", s)
    if not m:
        return None
    n = int(m.group(1))
    label = f"Q{n}"
    try:
        return ids.index(label)
    except ValueError:
        return n - 1


def go_to(i: int, active_indices: List[int]):
    st.session_state.local_idx = max(0, min(len(active_indices) - 1, i))


def load_progress(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "wrong_ids": set(data.get("wrong_ids", [])),
                    "attempts": int(data.get("attempts", 0)),
                    "correct": int(data.get("correct", 0)),
                    "seen_ids": set(data.get("seen_ids", [])),
                }
        except Exception:
            pass
    return {"wrong_ids": set(), "attempts": 0, "correct": 0, "seen_ids": set()}


def save_progress(path: str, progress: dict):
    data = {
        "wrong_ids": sorted(list(progress["wrong_ids"])),
        "attempts": int(progress["attempts"]),
        "correct": int(progress["correct"]),
        "seen_ids": sorted(list(progress["seen_ids"])),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def detect_choices_column(df) -> Optional[str]:
    for c in CHOICES_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def split_choices(val: str) -> List[str]:
    """Split only on '|' or ';' (never commas), preserving commas in numbers like $140,268.58"""
    s = str(val).strip()
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";")]
    else:
        parts = [s] if s else []
    return [p for p in parts if p]


def safe_equal(a: str, b: str) -> bool:
    a = html.unescape(str(a)).strip().lower()
    b = html.unescape(str(b)).strip().lower()
    return a == b


def mark_attempt(qid: str, correct: bool):
    st.session_state.progress["attempts"] += 1
    st.session_state.progress["seen_ids"].add(qid)
    if correct:
        st.session_state.progress["correct"] += 1
        st.session_state.progress["wrong_ids"].discard(qid)
    else:
        st.session_state.progress["wrong_ids"].add(qid)
    save_progress(PROGRESS_PATH, st.session_state.progress)


def reset_everything():
    try:
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
    except Exception:
        pass
    for k in [
        "always_show",
        "local_idx",
        "active_indices",
        "progress",
        "selection_signature",
        "show_answer_once",
        "last_qid",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


def robust_read_csv(path: str) -> pd.DataFrame:
    """Try multiple encodings & separators; handle messy CSVs."""
    attempts = [
        dict(encoding="utf-8-sig"),
        dict(encoding="utf-8"),
        dict(encoding="cp1252"),
        dict(encoding="latin1"),
        dict(encoding="utf-8-sig", engine="python"),
        dict(encoding="utf-8-sig", engine="python", sep=None),  # infer delimiter
        dict(encoding="utf-8-sig", engine="python", on_bad_lines="warn"),
    ]
    errs = []
    for kw in attempts:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            errs.append(f"{kw}: {e}")
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(path)
        except Exception as e:
            errs.append(f"excel: {e}")
    raise RuntimeError("\n\n".join(errs))


def parse_correct_index_from_text(val: str, num_options: int) -> Optional[int]:
    """Parse 'Correct Option: 4' or 'Correct Answer: D' -> zero-based index."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val)
    m = re.search(
        r"(?:Correct\s*(?:Option|Answer))\s*:\s*([A-Za-z]|\d+)", s, flags=re.IGNORECASE
    )
    if not m:
        return None
    tok = m.group(1).strip().upper()
    if tok.isdigit():
        k = int(tok) - 1
    else:
        k = ord(tok) - ord("A")
    return k if 0 <= k < num_options else None


def strip_correct_marker_and_format(text: str) -> str:
    """For showing full explanation from 'back': remove the marker and normalize <br> to line breaks."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    # remove a leading/first marker "Correct Option/Answer: X"
    s = re.sub(
        r"(?i)^\s*Correct\s*(?:Option|Answer)\s*:\s*[A-Za-z0-9]+\s*[-–—:]?\s*",
        "",
        s,
        count=1,
    )
    s = re.sub(r"(?i)Correct\s*(?:Option|Answer)\s*:\s*[A-Za-z0-9]+\s*", "", s, count=1)
    return s.strip()


def determine_correct_index(
    answer_val: str, back_val: str, options: List[str]
) -> Optional[int]:
    """Prefer parsing from answer; else from back; else match answer text to an option."""
    k = parse_correct_index_from_text(answer_val, len(options))
    if k is not None:
        return k
    k = parse_correct_index_from_text(back_val, len(options))
    if k is not None:
        return k
    if answer_val:
        s = str(answer_val).strip().lower()
        for i, opt in enumerate(options):
            if s == str(opt).strip().lower():
                return i
    return None


# ---------- LOAD DATA ----------
if not os.path.exists(CSV_PATH):
    st.error(f"CSV not found: {CSV_PATH}")
    st.stop()

try:
    df = robust_read_csv(CSV_PATH)
except Exception as e:
    st.error("Could not read CSV. Details:\n\n" + str(e))
    st.stop()

q_col = pick_col(df, QUESTION_COL_CANDIDATES)
a_col = pick_col(
    df, ANSWER_COL_CANDIDATES
)  # may be 'answer' or 'back' depending on file
choices_col = detect_choices_column(df)

# Remember 'back' explicitly for full explanations
back_col = "back" if "back" in df.columns else None

# Build ID labels from question text
ids = [extract_qnum_label(df[q_col].iloc[i], i) for i in range(len(df))]

# ---------- STATE ----------
if "always_show" not in st.session_state:
    st.session_state.always_show = False
if "local_idx" not in st.session_state:
    st.session_state.local_idx = 0
if "active_indices" not in st.session_state:
    st.session_state.active_indices = list(range(len(df)))
if "progress" not in st.session_state:
    st.session_state.progress = load_progress(PROGRESS_PATH)
if "selection_signature" not in st.session_state:
    st.session_state.selection_signature = ""

# ---------- SIDEBAR SETTINGS ----------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Choose set to practice",
    options=["All", "Range", "Random N", "Wrong only", "Not done yet"],
    help="Pick which questions to include in the current run.",
)

start_num = end_num = None
n_random = None

if mode == "Range":
    start_num = st.sidebar.number_input(
        "Start question #", min_value=1, value=1, step=1
    )
    end_num = st.sidebar.number_input(
        "End question #", min_value=1, value=len(df), step=1
    )
    if end_num < start_num:
        st.sidebar.warning("End must be ≥ Start.")
elif mode == "Random N":
    n_random = st.sidebar.number_input(
        "How many questions?",
        min_value=1,
        max_value=max(1, len(df)),
        value=min(20, len(df)),
        step=1,
    )

shuffle = st.sidebar.toggle(
    "Shuffle order", value=(mode in ["Random N", "Wrong only", "Not done yet"])
)
practice_wrong_only = mode == "Wrong only"
practice_not_done = mode == "Not done yet"

apply_btn = st.sidebar.button("Apply selection")

st.sidebar.divider()
if st.sidebar.button("🔄 Reset progress & settings"):
    reset_everything()


# ---------- BUILD SELECTION ----------
def build_active_indices() -> List[int]:
    if practice_wrong_only:
        wrong_set = st.session_state.progress["wrong_ids"]
        return [i for i, _ in enumerate(ids) if ids[i] in wrong_set]

    if practice_not_done:
        seen = st.session_state.progress["seen_ids"]
        return [i for i, _ in enumerate(ids) if ids[i] not in seen]

    if mode == "All":
        idxs = list(range(len(df)))
    elif mode == "Range":
        idxs = []
        if start_num is not None and end_num is not None:
            for qn in range(int(start_num), int(end_num) + 1):
                label = f"Q{qn}"
                if label in ids:
                    idxs.append(ids.index(label))
            if not idxs:
                lo = max(0, int(start_num) - 1)
                hi = min(len(df), int(end_num))
                idxs = list(range(lo, hi))
    elif mode == "Random N":
        k = int(n_random or 1)
        base = list(range(len(df)))
        idxs = random.sample(base, k=min(k, len(base)))
    else:
        idxs = list(range(len(df)))

    if shuffle:
        random.shuffle(idxs)
    return idxs


signature = f"{mode}-{start_num}-{end_num}-{n_random}-{shuffle}-{practice_wrong_only}-{practice_not_done}-{len(st.session_state.progress['wrong_ids'])}-{len(st.session_state.progress['seen_ids'])}"
if apply_btn or signature != st.session_state.selection_signature:
    st.session_state.active_indices = build_active_indices()
    st.session_state.local_idx = 0
    st.session_state.selection_signature = signature
    st.rerun()

active_indices = st.session_state.active_indices
if not active_indices:
    st.warning("No questions in the current selection. Adjust settings on the left.")
    st.stop()

# ---------- TOP BAR ----------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1, 2, 2, 1])

with c1:
    if st.button("⟵ Previous", use_container_width=True):
        go_to(st.session_state.local_idx - 1, active_indices)
        st.rerun()

with c2:
    with st.container():
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        target = st.text_input(
            "Jump to Q#", placeholder="e.g., 47 or Q47", label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    go = st.button("Go", use_container_width=True)
    if go:
        tgt_global = parse_target_index(target, ids)
        if (
            tgt_global is None
            or tgt_global < 0
            or tgt_global >= len(df)
            or tgt_global not in active_indices
        ):
            st.warning("Not found in current selection. Enter like 47 or Q47.")
        else:
            st.session_state.local_idx = active_indices.index(tgt_global)
            st.rerun()

with c3:
    st.session_state.always_show = st.toggle(
        "Always show answers",
        value=st.session_state.always_show,
        help="When ON, the answer/explanation panel displays automatically.",
    )

with c4:
    if st.button("Next ⟶", use_container_width=True):
        go_to(st.session_state.local_idx + 1, active_indices)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
global_idx = active_indices[st.session_state.local_idx]
q = df[q_col].iloc[global_idx]
a = df[a_col].iloc[global_idx]  # may be 'answer' or 'back' depending on pick_col
qid = ids[global_idx]

# Clear one-time reveal flag when question changes
if st.session_state.get("last_qid") != qid:
    st.session_state.show_answer_once = False
    st.session_state.last_qid = qid

st.markdown(f"### **{qid}**")
st.markdown(
    f"<div class='question-card'>{md_linebreaks(q)}</div>", unsafe_allow_html=True
)

feedback_msg = ""
was_correct: Optional[bool] = None

# Prefer 'back' for explanation (full text), even if 'answer' exists
raw_expl = df[back_col].iloc[global_idx] if back_col else a
expl = strip_correct_marker_and_format(raw_expl)

has_mcq = bool(choices_col and pd.notna(df[choices_col].iloc[global_idx]))

if has_mcq:
    # --- MCQ flow: choices + click/submit ---
    options = split_choices(df[choices_col].iloc[global_idx])
    options_numbered = [f"{i+1}. {opt}" for i, opt in enumerate(options)]
    choice_display = st.radio("Choose an answer:", options=options_numbered, index=None)

    # Determine correct index (prefer 'answer', else 'back', else text match)
    # Use the *answer* column content if present in df; else the picked a_col
    answer_text = df["answer"].iloc[global_idx] if "answer" in df.columns else a
    back_text = df["back"].iloc[global_idx] if "back" in df.columns else raw_expl
    correct_idx0 = determine_correct_index(answer_text, back_text, options)

    submitted = st.button("Submit")

    if submitted:
        selected_idx0 = (
            options_numbered.index(choice_display)
            if choice_display is not None
            else None
        )
        is_correct = False
        if correct_idx0 is not None and selected_idx0 is not None:
            is_correct = selected_idx0 == correct_idx0
        elif selected_idx0 is not None and correct_idx0 is None:
            # fallback to comparing selected text to 'a'
            is_correct = safe_equal(options[selected_idx0], a)

        mark_attempt(qid, bool(is_correct))
        was_correct = bool(is_correct)
        feedback_msg = "✅ Correct!" if is_correct else "❌ Not quite."

    # Show the correct option + full explanation if submitted or always_show
    if (submitted or st.session_state.always_show) and options:
        if correct_idx0 is not None:
            correct_text = options[correct_idx0]
            st.markdown(
                f"<div class='answer'><b>Answer</b><br>"
                f"Correct option: <span class='badge'>{correct_idx0+1}</span><br>"
                f"{md_linebreaks(correct_text)}"
                + (
                    f"<hr style='opacity:.25;margin:8px 0'>{md_linebreaks(expl)}"
                    if expl
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='answer'><b>Answer</b><br>{md_linebreaks(expl or a)}</div>",
                unsafe_allow_html=True,
            )

else:
    # --- Non-MCQ fallback (if any rows lack choices) ---
    if (
        st.session_state.always_show
        or st.button("Reveal answer")
        or st.session_state.get("show_answer_once")
    ):
        st.session_state.show_answer_once = True
        st.markdown(
            f"<div class='answer'><b>Answer</b><br>{md_linebreaks(expl or a)}</div>",
            unsafe_allow_html=True,
        )
        col_ok, col_bad = st.columns(2)
        with col_ok:
            if st.button("I was right"):
                mark_attempt(qid, True)
                st.session_state.show_answer_once = False
                st.rerun()
        with col_bad:
            if st.button("I was wrong"):
                mark_attempt(qid, False)
                st.session_state.show_answer_once = False
                st.rerun()

# Feedback line for MCQ
if feedback_msg:
    st.markdown(f"<div class='feedback'>{feedback_msg}</div>", unsafe_allow_html=True)

# ---------- FOOTER: PAGER + STATS ----------
st.caption(
    f"{st.session_state.local_idx+1} / {len(active_indices)}  •  Global: {global_idx+1} / {len(df)}"
)

att = st.session_state.progress["attempts"]
cor = st.session_state.progress["correct"]
acc = f"{(100*cor/att):.1f}%" if att else "—"
wrong_count = len(st.session_state.progress["wrong_ids"])
seen_count = len(st.session_state.progress["seen_ids"])
not_done = max(0, len(df) - seen_count)

st.caption(
    f"Attempts: {att}  •  Correct: {cor}  •  Accuracy: {acc}  •  Wrong saved: {wrong_count}  •  "
    f"Done: {seen_count}/{len(df)}  •  Not done yet: {not_done}"
)

# Next only (no add/remove wrong buttons)
if st.button("Next ▶"):
    go_to(st.session_state.local_idx + 1, active_indices)
    st.rerun()
