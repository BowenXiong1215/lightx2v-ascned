"""Upgrade an existing Ascend patch install without replacing local trainer fixes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parent
REPLACEMENTS = PATCH_ROOT / "payload" / "replacements"
ADDITIONS = PATCH_ROOT / "payload" / "additions"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_checked(target_root: Path, relative: str, accepted: set[str]) -> None:
    target = target_root / relative
    source = REPLACEMENTS / relative
    if not target.is_file() or not source.is_file():
        raise SystemExit(f"Missing source or target for {relative}")
    current = sha256(target)
    desired = sha256(source)
    if current == desired:
        print(f"already current: {relative}")
        return
    if current not in accepted:
        raise SystemExit(f"Refusing to replace locally modified file: {relative} sha256={current}")
    shutil.copy2(target, target.with_suffix(target.suffix + ".pre_official_lora_init"))
    shutil.copy2(source, target)
    print(f"updated: {relative}")


def patch_trainer(target_root: Path) -> None:
    relative = "lightx2v_train/lightx2v_train/trainers/dmd/trainer.py"
    target = target_root / relative
    text = target.read_text()
    import_line = "from lightx2v_train.runtime.gradients import clip_grad_norm_\n"
    if import_line not in text:
        anchor = "from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value\n"
        if anchor not in text:
            raise SystemExit(f"Cannot find DMD trainer import anchor in {target}")
        text = text.replace(anchor, anchor + import_line, 1)
    calls = text.count("torch.nn.utils.clip_grad_norm_(")
    if calls:
        text = text.replace("torch.nn.utils.clip_grad_norm_(", "clip_grad_norm_(")
    elif "clip_grad_norm_(" not in text:
        raise SystemExit(f"Cannot find gradient clipping calls in {target}")
    backup = target.with_suffix(target.suffix + ".pre_official_lora_init")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(text)
    print(f"patched in place: {relative} calls={calls}")


def install_gradient_helper(target_root: Path) -> None:
    relative = "lightx2v_train/lightx2v_train/runtime/gradients.py"
    target = target_root / relative
    source = ADDITIONS / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != sha256(source):
        raise SystemExit(f"Existing gradient helper differs: {target}")
    shutil.copy2(source, target)
    print(f"installed: {relative}")


def compile_sources(target_root: Path) -> None:
    files = [
        "lightx2v_train/lightx2v_train/model_zoo/base.py",
        "lightx2v_train/lightx2v_train/trainers/dmd/runtime.py",
        "lightx2v_train/lightx2v_train/trainers/dmd/trainer.py",
        "lightx2v_train/lightx2v_train/runtime/gradients.py",
    ]
    for relative in files:
        path = target_root / relative
        compile(path.read_text(), str(path), "exec")
    print("Python syntax verification: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="LightX2V source root")
    args = parser.parse_args()
    target = args.target.resolve()
    if not (target / "pyproject.toml").is_file():
        raise SystemExit(f"Not a LightX2V source root: {target}")

    replace_checked(
        target,
        "lightx2v_train/lightx2v_train/model_zoo/base.py",
        {
            "5200bfd5c4cc054abf82ea11a3b4d326ecd57dacb43439c5e2ded253d511352b",
            "2feec0bd3c90bb0ff74fbcced53e49f4517abb9c375466dfeee887f4f10a5725",
        },
    )
    replace_checked(
        target,
        "lightx2v_train/lightx2v_train/trainers/dmd/runtime.py",
        {"accf35154346264c26db44ee0d8291d1c6e334f0d5aacca144d3be6381704138"},
    )
    install_gradient_helper(target)
    patch_trainer(target)
    compile_sources(target)
    print("Official LoRA warm-start and bounded NPU clipping upgrade: PASS")


if __name__ == "__main__":
    main()
