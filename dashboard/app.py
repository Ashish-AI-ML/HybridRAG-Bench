"""
dashboard/app.py — Streamlit Evaluation Dashboard for HybridRAG Bench.

Provides a visual interface for:
  1. Interactive query testing with source traceability
  2. Evaluation results explorer (per-query metrics)
  3. Ablation comparison chart (BM25 vs Dense vs Hybrid vs +Reranker)
  4. Latency profiling breakdown
  5. Cost estimation projections

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import get_config
from src.pipeline import HybridRAGPipeline

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HybridRAG Bench",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin: 0.5rem 0 0; }

    .metric-card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
    .metric-card .label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    .chunk-card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.75rem;
    }
    .chunk-card .meta { font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem; }
    .chunk-card .text { font-size: 0.87rem; color: #e2e8f0; line-height: 1.6; }
    .score-badge {
        display: inline-block; background: #1e3a5f; color: #38bdf8;
        padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; margin-right: 4px;
    }
    .answer-box {
        background: #0f172a; border-left: 3px solid #38bdf8;
        padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
    .confidence-high { color: #22c55e; }
    .confidence-mid  { color: #f59e0b; }
    .confidence-low  { color: #ef4444; }
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 2rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .badge-mode {
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
        background: #1e293b; color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline caching ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Building index (first run only)...")
def load_pipeline():
    cfg = get_config()
    pipe = HybridRAGPipeline(cfg=cfg)
    pipe.build_index()
    return pipe


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        ["hybrid_rerank", "hybrid", "dense_only", "bm25_only"],
        index=0,
        help="Ablation mode: compare retrieval strategies"
    )
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)
    strict_mode = st.toggle("Strict Grounding", value=True,
                            help="Force LLM to answer ONLY from retrieved context")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")
    results_dir = Path("data/eval_results")
    result_files = sorted(results_dir.glob("eval_*.json"), reverse=True) if results_dir.exists() else []

    if result_files:
        selected_file = st.selectbox(
            "Load Results File",
            [f.name for f in result_files],
            index=0,
        )
        show_results = st.button("📂 Load Results")
    else:
        st.info("No evaluation results yet.\nRun: `python -m src.evaluation.suite_runner`")
        show_results = False
        selected_file = None

    st.markdown("---")
    st.markdown("**HybridRAG Bench v2.0**")
    st.markdown("BM25 + Qdrant + Reranker")


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 HybridRAG Bench</h1>
    <p>Production-grade Hybrid Retrieval-Augmented Generation · BM25 + Qdrant + Cross-Encoder Reranking</p>
</div>
""", unsafe_allow_html=True)

tab_query, tab_eval, tab_ablation = st.tabs(["🔍 Query", "📊 Evaluation Results", "🧪 Ablation Analysis"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Query Interface
# ─────────────────────────────────────────────────────────────────────────────
with tab_query:
    col_q, col_mode = st.columns([4, 1])
    with col_q:
        question = st.text_area(
            "Ask a question",
            placeholder="e.g. Why did Shor's algorithm threaten RSA encryption?",
            height=80,
        )
    with col_mode:
        st.markdown(f"<br><span class='badge-mode'>Mode: {retrieval_mode}</span>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Query", use_container_width=True)

    # Sample questions
    with st.expander("💡 Sample Questions"):
        samples = [
            "What is Shor's algorithm and why is it a threat to RSA?",
            "How did Grover's algorithm achieve a quadratic speedup?",
            "How did Deutsch's 1985 work build on Feynman's 1981 proposal?",
            "What did Google's Sycamore processor demonstrate in 2019?",
            "Who proved the quantum no-cloning theorem and when?",
        ]
        for s in samples:
            if st.button(s, key=s):
                question = s
                run_btn = True

    if run_btn and question.strip():
        with st.spinner(f"Running {retrieval_mode} pipeline..."):
            pipeline = load_pipeline()
            result = pipeline.query(question, top_k=top_k, mode=retrieval_mode, strict=strict_mode)

        # ── Answer ───────────────────────────────────────────────────────
        conf = result["confidence_score"]
        conf_class = "confidence-high" if conf > 0.7 else ("confidence-mid" if conf > 0.4 else "confidence-low")
        conf_icon = "🟢" if conf > 0.7 else ("🟡" if conf > 0.4 else "🔴")

        col_a, col_c = st.columns([5, 1])
        with col_a:
            st.markdown("#### 🤖 Generated Answer")
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value {conf_class}">{conf_icon} {conf:.2f}</div>
                <div class="label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        if result["is_insufficient"]:
            st.warning("⚠️ The system flagged this as INSUFFICIENT CONTEXT — answer may be unreliable.")

        # ── Latency breakdown ─────────────────────────────────────────────
        lat = result["latency"]
        st.markdown("#### ⏱ Latency Breakdown")
        lat_cols = st.columns(len(lat))
        for i, (stage, ms) in enumerate(lat.items()):
            if ms is not None:
                with lat_cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value" style="font-size:1.2rem">{ms:.0f}ms</div>
                        <div class="label">{stage.replace('_', ' ')}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Retrieved chunks (traceability) ──────────────────────────────
        st.markdown(f"#### 📄 Source Traceability ({len(result['retrieved_chunks'])} chunks)")
        for c in result["retrieved_chunks"]:
            scores = []
            if c.get("reranker_score") is not None:
                scores.append(f"<span class='score-badge'>reranker: {c['reranker_score']:.3f}</span>")
            if c.get("dense_score") is not None:
                scores.append(f"<span class='score-badge'>dense: {c['dense_score']:.3f}</span>")
            if c.get("bm25_score") is not None:
                scores.append(f"<span class='score-badge'>bm25: {c['bm25_score']:.2f}</span>")
            if c.get("rrf_score") is not None:
                scores.append(f"<span class='score-badge'>rrf: {c['rrf_score']:.4f}</span>")

            st.markdown(f"""
            <div class="chunk-card">
                <div class="meta">
                    📁 <b>{c['doc_id']}</b> &nbsp;·&nbsp; Rank #{c.get('final_rank', '?')}
                    &nbsp;&nbsp; {"".join(scores)}
                </div>
                <div class="text">{c['text']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Token usage
        tu = result.get("token_usage", {})
        if tu:
            st.caption(f"🪙 Token usage — input: {tu.get('input', 0)}, output: {tu.get('output', 0)}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Evaluation Results Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    if show_results and selected_file:
        result_path = results_dir / selected_file
        data = json.loads(result_path.read_text(encoding="utf-8"))

        if "per_query_results" in data:
            meta = data.get("run_metadata", {})
            agg = data.get("aggregate_metrics", {})
            lat_stats = data.get("latency_stats", {})
            cost = data.get("cost_summary", {})

            st.markdown(f"### Results: `{selected_file}`")
            st.caption(f"Mode: **{meta.get('mode')}** | Provider: **{meta.get('provider')}** | Queries: **{meta.get('n_queries')}**")

            # Aggregate metrics grid
            key_metrics = [
                ("mean_precision_at_1", "P@1"),
                ("mean_mrr", "MRR"),
                ("mean_ndcg_at_5", "NDCG@5"),
                ("mean_semantic_similarity", "Semantic Sim"),
                ("mean_faithfulness_score", "Faithfulness"),
                ("mean_entity_recall", "Entity Recall"),
            ]
            cols = st.columns(len(key_metrics))
            for i, (k, label) in enumerate(key_metrics):
                val = agg.get(k)
                if val is not None:
                    with cols[i]:
                        st.metric(label, f"{val:.3f}")

            # Per-query table
            st.markdown("#### Per-Query Results")
            df = pd.DataFrame(data["per_query_results"])
            display_cols = [
                "q_id", "q_type", "confidence_score",
                "precision_at_1", "mrr", "semantic_similarity",
                "faithfulness_score", "entity_recall", "latency_ms"
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].style.format({c: "{:.3f}" for c in display_cols if c not in ("q_id", "q_type")}),
                use_container_width=True,
            )

            # Latency distribution
            if "latency_ms" in df.columns:
                st.markdown("#### Latency Distribution")
                fig = px.histogram(df, x="latency_ms", nbins=20,
                                   title="Query Latency Distribution (ms)",
                                   color_discrete_sequence=["#38bdf8"])
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a",
                                  plot_bgcolor="#0f172a")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Select an evaluation results file from the sidebar and click 'Load Results'.")
        st.markdown("""
        **How to generate results:**
        ```bash
        # Standard evaluation
        python -m src.evaluation.suite_runner

        # Full ablation study
        python -m src.evaluation.suite_runner --ablation
        ```
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Ablation Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_ablation:
    ablation_files = sorted(results_dir.glob("eval_*.json"), reverse=True) if results_dir.exists() else []
    ablation_data_all = []

    for f in ablation_files[:5]:  # compare up to 5 most recent runs
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if "run_metadata" in d:
                agg = d.get("aggregate_metrics", {})
                mode = d["run_metadata"].get("mode", "unknown")
                ablation_data_all.append({"mode": mode, "file": f.name, **agg})
            elif "ablation" in d:
                # Format from `run_evaluation --ablation`
                for abl_mode, agg in d["ablation"].items():
                    # only keep mean_ metrics to avoid clutter
                    agg_metrics = {k: v for k, v in agg.items() if k.startswith("mean_")}
                    # ensure we don't duplicate mode runs if there are multiple files
                    if not any(x["mode"] == abl_mode for x in ablation_data_all):
                        ablation_data_all.append({"mode": abl_mode, "file": f.name, **agg_metrics})
        except Exception:
            pass

    if ablation_data_all:
        df_abl = pd.DataFrame(ablation_data_all)
        metric_choices = [c for c in df_abl.columns if c.startswith("mean_") and df_abl[c].notna().any()]

        st.markdown("#### Retrieval Strategy Comparison")
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            metric_choices,
            default=metric_choices[:4] if len(metric_choices) >= 4 else metric_choices,
        )

        if selected_metrics:
            fig = go.Figure()
            for idx, row in df_abl.iterrows():
                fig.add_trace(go.Bar(
                    name=row["mode"],
                    x=selected_metrics,
                    y=[row.get(m, 0) for m in selected_metrics],
                ))
            fig.update_layout(
                barmode="group",
                title="Retrieval Mode Comparison",
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                legend_title="Mode",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Raw Comparison Table")
        st.dataframe(df_abl[["mode", "file"] + metric_choices[:8]], use_container_width=True)
    else:
        st.info("No evaluation result files found. Run the evaluation suite to populate data.")
        st.markdown("""
        ```bash
        python -m src.evaluation.suite_runner --ablation
        ```
        This runs all 4 retrieval modes and saves timestamped results for comparison.
        """)
