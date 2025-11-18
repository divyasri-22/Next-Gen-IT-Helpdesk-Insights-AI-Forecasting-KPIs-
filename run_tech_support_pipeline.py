from pathlib import Path
import yaml

from tech_support_pipeline.tasks import (
    load_tickets,
    enrich_tickets,
    summarize_tickets,
    save_summary,
)


def main():
    # Project root: the folder where THIS file lives
    project_root = Path(__file__).parent

    # Config lives inside the tech_support_pipeline folder
    config_path = project_root / "tech_support_pipeline" / "config.yaml"
    print(f"➡️ Using config file: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load YAML config
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print("Raw config loaded from YAML:", config)

    context = config.get("context", {})
    if "input_path" not in context or "output_path" not in context:
        raise KeyError("config must have context.input_path and context.output_path")

    # Make paths absolute relative to project_root
    context["input_path"] = str(project_root / context["input_path"])
    context["output_path"] = str(project_root / context["output_path"])

    print("Resolved context:", context)

    # Run the pipeline
    print("▶️ Step 1: load_tickets()")
    df = load_tickets(context)
    print(f"   Loaded {len(df)} tickets")

    print("▶️ Step 2: enrich_tickets()")
    df = enrich_tickets(context, df)
    print(f"   After enrichment: {len(df)} tickets")

    print("▶️ Step 3: summarize_tickets()")
    summary_df = summarize_tickets(context, df)
    print(f"   Summary has {len(summary_df)} rows")

    print("▶️ Step 4: save_summary()")
    save_summary(context, summary_df)

    print("✅ Tech support pipeline finished successfully!")
    print(f"✅ Summary saved to: {context['output_path']}")


if __name__ == "__main__":
    main()
