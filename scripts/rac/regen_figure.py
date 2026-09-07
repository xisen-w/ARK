"""Regenerate one concept figure through PaperBanana, grounded in the real method.

The in-run figures for the finance arm were drawn by PaperBanana, then
overwritten by the writer agent with a matplotlib diagram after the reviewer
called the AI figures wrong for the domain. This runs the pipeline again with
the method text as context and the pro image tier, so the result can be judged
on its merits before anything in the paper changes.

    python scripts/rac/regen_figure.py --project rac_finance_b --name fig_overview_pro \
        --caption "..." --context-file /tmp/method.txt
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import pathlib
import sys

import yaml

ARK = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--caption", required=True)
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--text-model", default="gemini-3.1-pro-preview")
    ap.add_argument("--image-model", default="gemini-3-pro-image")
    args = ap.parse_args()

    cfg = yaml.safe_load((ARK / ".ark/config.yaml").read_text())
    os.environ["OPENROUTER_API_KEY"] = cfg["openrouter_api_key"]
    os.environ["MAIN_MODEL_NAME"] = args.text_model
    os.environ["IMAGE_GEN_MODEL_NAME"] = args.image_model
    # Keep the Gemini-direct path out of the way: only OpenRouter is funded.
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(var, None)

    pb = ARK / "submodules/PaperBanana"
    sys.path.insert(0, str(pb))
    from agents.critic_agent import CriticAgent
    from agents.planner_agent import PlannerAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.stylist_agent import StylistAgent
    from agents.visualizer_agent import VisualizerAgent
    from utils.config import ExpConfig
    from utils.paperviz_processor import PaperVizProcessor

    data_dir = pb / "data" / "PaperBananaBench"
    retrieval = "auto" if (data_dir / "diagram" / "ref.json").exists() else "none"
    exp_config = ExpConfig(
        dataset_name="PaperBananaBench", task_name="diagram", exp_mode="demo_full",
        retrieval_setting=retrieval, max_critic_rounds=3, work_dir=pb,
        main_model_name=args.text_model, image_gen_model_name=args.image_model,
    )
    processor = PaperVizProcessor(
        exp_config=exp_config, vanilla_agent=None,
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=None,
    )
    data = {
        "candidate_id": args.name,
        "content": pathlib.Path(args.context_file).read_text(),
        "visual_intent": (
            f"{args.caption} STYLE: Labels MAX 3-5 words, NO sentences inside "
            "components. BUT make icons detailed and elaborate (not simple flat "
            "shapes). Layout should be COMPACT — minimize whitespace, pack "
            "components closely. The figure should feel dense and "
            "information-rich through its visual elements, not through text."
        ),
        "additional_info": {"rounded_ratio": args.aspect},
    }
    print(f"running PaperBanana: text={args.text_model} image={args.image_model} "
          f"retrieval={retrieval}", flush=True)
    result = asyncio.run(processor.process_single_query(data, do_eval=False))
    out = ARK / "projects" / args.project / "paper/figures" / f"{args.name}.png"
    for key in sorted(result.keys(), reverse=True):
        if "base64_jpg" in key and result[key] and len(result[key]) > 100:
            out.write_bytes(base64.b64decode(result[key]))
            print(f"wrote {out} ({out.stat().st_size} bytes)")
            return 0
    print("pipeline produced no image; keys:", sorted(result.keys())[:12])
    return 1


if __name__ == "__main__":
    sys.exit(main())
