from pathlib import Path
import yaml

from tech_support_pipeline.tasks import (
    load_tickets,
    normalize_columns,
    tag_issue_type,
    compute_sla_metrics,
    save_outputs,
)


def main() -> None:
    project_root = Path(__file__).parent

    # 1. Load config
    config_path = project_root / "tech_support_pipeline" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    print(f"➡️ Using config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    print("Raw config loaded from YAML:", raw_config)

    context = raw_config.get("context", {})
    if "input_path" not in context or "output_dir" not in context:
        raise KeyError("config.context must have input_path and output_dir")

    # Make paths absolute
    context["input_path"] = str(project_root / Path(context["input_path"]))
    context["output_dir"] = str(project_root / Path(context["output_dir"]))

    print("Resolved context:", context)

    # 2. Run pipeline steps
    print("▶️ Step 1: load_tickets()")
    df = load_tickets(context)
    print(f"   Loaded {len(df)} tickets")

    print("▶️ Step 2: normalize_columns()")
    df = normalize_columns(context, df)
    print("   Columns normalised")

    print("▶️ Step 3: tag_issue_type()")
    df = tag_issue_type(context, df)
    print("   Issue types tagged")

    print("▶️ Step 4: compute_sla_metrics()")
    enriched_df, metrics_by_type, metrics_by_agent = compute_sla_metrics(context, df)
    print("   SLA metrics computed")

    print("▶️ Step 5: save_outputs()")
    save_outputs(context, enriched_df, metrics_by_type, metrics_by_agent)
    print("✅ Tech support pipeline finished!")
    print(f"✅ Outputs written under: {context['output_dir']}")


if __name__ == "__main__":
    main()
