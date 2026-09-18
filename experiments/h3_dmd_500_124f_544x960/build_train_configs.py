"""Derive a fresh 124-frame H3 DMD probe and 500-step run from a known-good config."""

import argparse
import copy
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/hpc-to-ds-0115/x00876811/light/h3_dmd_overnight.yaml"),
        help="Existing working H3 DMD training config on the server",
    )
    args = parser.parse_args()
    package_dir = Path(__file__).resolve().parent
    prompts = package_dir / "h3_train_prompts.txt"
    if len([line for line in prompts.read_text().splitlines() if line.strip()]) != 300:
        raise SystemExit("Expected exactly 300 nonblank training prompts")
    source = yaml.safe_load(args.source.read_text())
    if source["model"]["name"] != "minimax_h3_t2av" or source["training"]["method"] != "dmd":
        raise SystemExit("Source config is not an H3 DMD training config")
    if source["data"]["train"]["name"] != "prompt_dataset":
        raise SystemExit("Source config does not use prompt_dataset")

    train = copy.deepcopy(source)
    train["data"]["train"]["data_path"] = [str(prompts)]
    train["training"]["max_train_iters"] = 500
    train["training"]["save_every_iters"] = 100
    train["training"]["save_total_limit"] = 6
    train["training"]["dmd"]["num_inference_steps"] = 4
    train["training"]["dmd"]["generation_shapes"] = [{"value": [124, 544, 960]}]
    dm = train["model"]["capabilities"]["distribution_matching"]
    dm["video_flow_shift"] = 12.0
    dm["audio_flow_shift"] = 3.0
    train["training"]["output_dir"] = str(package_dir / "output_500")
    train.setdefault("resume", {})["auto_resume"] = False
    train["training"]["student"].pop("checkpoint_path", None)

    probe = copy.deepcopy(train)
    probe["training"]["max_train_iters"] = 1
    probe["training"]["save_every_iters"] = 1
    probe["training"]["output_dir"] = str(package_dir / "output_probe")
    for name, config in (("h3_probe_1step.yaml", probe), ("h3_train_500step.yaml", train)):
        path = package_dir / name
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        print(path)


if __name__ == "__main__":
    main()
