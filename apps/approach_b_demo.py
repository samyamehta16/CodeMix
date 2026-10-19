"""
Approach B — simple Streamlit demo for Tanglish claim verification.

Run from project root (needs GEMINI_API_KEY + SERPER_API_KEY in .env):

    pip install streamlit
    streamlit run apps/approach_b_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "verification"))
sys.path.insert(0, str(ROOT / "src"))

from pipeline import (  # noqa: E402
    STAGE_COMPLETE,
    STAGE_LLM_UNAVAILABLE,
    STAGE_NO_CLAIM,
    STAGE_NO_RESULTS,
    STAGE_SEARCH_UNAVAILABLE,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# 5 prepared demo claims (expected class is for the presenter, not the model)
# ---------------------------------------------------------------------------
DEMO_CLAIMS = [
    {
        "label": "1 · FAKE",
        "expected": "FAKE",
        "why": "Viral ₹5000-note rumor; contradicted by RBI / PIB fact-checks.",
        "text": "RBI 5000 rupees note release pannitanga nu news la varudhu, real ah fake ah verify pannunga.",
    },
    {
        "label": "2 · REAL",
        "expected": "REAL",
        "why": "Chandrayaan-3 soft landing is publicly confirmed by ISRO.",
        "text": "Chandrayaan 3 moon la soft land aayiduchu nu ISRO solli irukanga.",
    },
    {
        "label": "3 · MISLEADING",
        "expected": "MISLEADING",
        "why": "Often only some districts get holidays — statewide wording overclaims.",
        "text": "TN heavy rain school holiday collector declare pannitara, ella school um holiday dhaan.",
    },
    {
        "label": "4 · UNVERIFIED",
        "expected": "UNVERIFIED",
        "why": "Private / unpublished claim — search usually finds no checkable evidence.",
        "text": "Naan ketten nu oru WhatsApp forward la solraanga, next week Chennai la free iPhone distribution government pannum amma.",
    },
    {
        "label": "5 · HARD",
        "expected": "HARD / ambiguous",
        "why": (
            "Hard because: (a) timing/scope may change with fresh news, "
            "(b) 'free bus' schemes exist but details vary by state/year, "
            "(c) Tanglish + policy nuance → easy to get MISLEADING vs REAL wrong."
        ),
        "text": "Tamil Nadu la women ku bus free ah iruku forever nu government announce pannitanga nu viral aaguthu.",
    },
]

VERDICT_COLORS = {
    "REAL": "#1b7a3d",
    "FAKE": "#b00020",
    "MISLEADING": "#b36b00",
    "UNVERIFIED": "#555555",
}


def _init_state():
    if "claim_text" not in st.session_state:
        st.session_state.claim_text = DEMO_CLAIMS[0]["text"]
    if "active_demo" not in st.session_state:
        st.session_state.active_demo = DEMO_CLAIMS[0]
    if "result" not in st.session_state:
        st.session_state.result = None


def _set_demo(idx: int):
    demo = DEMO_CLAIMS[idx]
    st.session_state.claim_text = demo["text"]
    st.session_state.active_demo = demo
    st.session_state.result = None


def render_what_is():
    st.markdown(
        """
### What is Approach B?

**Approach B** is TruthGuard’s *evidence-grounded* track for **Tanglish**
(Tamil–English code-mixed) social claims. It does **not** fine-tune a classifier.
Instead it runs a fixed 6-stage pipeline:

1. **Claim extraction** — turn the Tanglish post into a clear checkable proposition  
2. **Query generation** — build short search queries  
3. **Web search** (Serper) — retrieve candidate pages  
4. **Page extraction** (Trafilatura) — pull readable text  
5. **Ranking** — score by relevance + domain authority + freshness  
6. **LLM verdict** (Gemini) — decide **only from retrieved evidence**

Modules 1–5 never say true/false. Only stage 6 outputs
`REAL` / `FAKE` / `MISLEADING` / `UNVERIFIED`, and it must **cite evidence IDs**.
Missing evidence → **UNVERIFIED**, never silently **FAKE**.
"""
    )


def render_result(result):
    st.subheader("Result")

    if result.stage_reached != STAGE_COMPLETE or result.verdict is None:
        st.warning(
            f"Pipeline stopped early (`{result.stage_reached}`). "
            + (" ".join(result.notes) if result.notes else "")
        )
        if result.claim and result.claim.normalized_claim:
            st.caption(f"Normalized claim: {result.claim.normalized_claim}")
        return

    v = result.verdict
    color = VERDICT_COLORS.get(v.verdict, "#333")
    st.markdown(
        f"""
<div style="border:2px solid {color}; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem;">
  <div style="font-size:0.85rem; color:#666; text-transform:uppercase; letter-spacing:0.04em;">Verdict</div>
  <div style="font-size:1.8rem; font-weight:700; color:{color};">{v.verdict}</div>
  <div style="margin-top:0.35rem; color:#444;">
    Relation: <b>{v.evidence_relation}</b> &nbsp;·&nbsp; Confidence: <b>{v.confidence:.2f}</b>
    &nbsp;·&nbsp; Cited: <b>{', '.join(v.evidence_ids) if v.evidence_ids else '—'}</b>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("**Explanation**")
    st.write(v.explanation or "—")

    st.markdown("**Cited sources**")
    cited = set(v.evidence_ids or [])
    shown = False
    for r in result.ranked_evidence:
        if cited and r.evidence_id not in cited:
            continue
        shown = True
        ev = r.evidence
        title = ev.result.title or "(no title)"
        url = ev.result.url
        domain = ev.result.source_domain
        st.markdown(
            f"- **{r.evidence_id}** — [{title}]({url})  \n"
            f"  `{domain}` · score={r.final_score:.2f} "
            f"(rel={r.relevance_score:.2f}, auth={r.authority_score:.2f}, fresh={r.freshness_score:.2f})"
        )
    if not shown:
        # fall back: show top ranked even if IDs didn't match
        for r in result.ranked_evidence[:5]:
            ev = r.evidence
            st.markdown(
                f"- **{r.evidence_id}** — [{ev.result.title}]({ev.result.url}) (`{ev.result.source_domain}`)"
            )

    with st.expander("Pipeline details (normalized claim, queries, notes)"):
        st.write("**Normalized claim:**", result.claim.normalized_claim)
        st.write("**Queries:**")
        for q in result.queries:
            st.write(f"- {q}")
        if result.notes:
            st.write("**Notes:**", result.notes)


def main():
    st.set_page_config(page_title="TruthGuard Approach B", page_icon="🔎", layout="centered")
    _init_state()

    st.title("TruthGuard · Approach B")
    st.caption("Evidence-grounded Tanglish claim verification (live web + Gemini)")

    render_what_is()
    st.divider()

    st.markdown("### Prepared demo claims")
    st.caption("Click one to load it, then press **Verify claim**.")
    cols = st.columns(5)
    for i, demo in enumerate(DEMO_CLAIMS):
        with cols[i]:
            if st.button(demo["label"], use_container_width=True, key=f"demo_{i}"):
                _set_demo(i)

    active = st.session_state.active_demo
    st.info(f"**Presenter note ({active['expected']}):** {active['why']}")

    st.markdown("### Claim input")
    text = st.text_area(
        "Tanglish / English claim",
        key="claim_text",
        height=100,
        placeholder="Type or paste a Tanglish claim…",
    )

    run = st.button("Verify claim", type="primary", use_container_width=True)

    if run:
        if not text.strip():
            st.error("Enter a claim first.")
        else:
            with st.spinner("Running Approach B pipeline (search + extract + verify)…"):
                try:
                    st.session_state.result = run_pipeline(text.strip())
                except Exception as e:
                    st.session_state.result = None
                    st.error(f"Pipeline error: {e}")

    if st.session_state.result is not None:
        st.divider()
        render_result(st.session_state.result)


if __name__ == "__main__":
    main()
