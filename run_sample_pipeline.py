from pathlib import Path
import yaml
from sample_pipeline.tasks import load_raw_data, clean_data, save_output


def main():
    # Project root is the folder where this file lives
    project_root = Path(__file__).parent

    # Our config is: project_root / sample_pipeline / config.yaml
    config_path = project_root / "sample_pipeline" / "config.yaml"
    print(f"➡️ Using config file: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load YAML
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print("Raw config loaded from YAML:", config)

    if "context" not in config:
        raise KeyError("config must have a top-level key 'context'")

    context = config["context"]

    # Validate required keys
    if "input_path" not in context or "output_path" not in context:
        raise KeyError(
            "config['context'] must have 'input_path' and 'output_path'"
        )

    # Resolve relative paths to absolute paths
    input_path = (project_root / context["input_path"]).resolve()
    output_path = (project_root / context["output_path"]).resolve()

    context["input_path"] = str(input_path)
    context["output_path"] = str(output_path)

    print("Resolved context:", context)

    # Run the tasks
    print("▶️ Step 1: load_raw_data()")
    df = load_raw_data(context)
    print(f"   Loaded {len(df)} rows")

    print("▶️ Step 2: clean_data()")
    df = clean_data(context, df)
    print(f"   After cleaning: {len(df)} rows")

    print("▶️ Step 3: save_output()")
    save_output(context, df)
    print("✅ Pipeline finished successfully!")
    print(f"✅ Output saved to: {output_path}")


if __name__ == "__main__":
    main()
