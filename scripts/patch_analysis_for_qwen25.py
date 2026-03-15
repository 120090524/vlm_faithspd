from __future__ import annotations

from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parent.parent

FILES_TO_COPY = {
    REPO_ROOT / "analysis" / "layer_analysis_generation.py": REPO_ROOT / "scripts" / "layer_analysis_generation_qwen25.py",
    REPO_ROOT / "analysis" / "analyze_layer_changes.py": REPO_ROOT / "scripts" / "analyze_layer_changes_qwen25.py",
    REPO_ROOT / "analysis" / "neuron_activation.py": REPO_ROOT / "scripts" / "neuron_activation_qwen25.py",
}

UTILS_FILE = REPO_ROOT / "utils" / "model_utils.py"
UTILS_BACKUP = REPO_ROOT / "utils" / "model_utils.py.bak_qwen2vl"


def patch_text(text: str) -> str:
    replacements = [
        ("Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
        ("from mechanism_analysis.tam import TAM", "from analysis.tam import TAM"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_utils_model_utils() -> None:
    if not UTILS_FILE.exists():
        print(f"[Skip] utils file not found: {UTILS_FILE}")
        return

    if not UTILS_BACKUP.exists():
        shutil.copy2(UTILS_FILE, UTILS_BACKUP)
        print(f"[OK] backup created: {UTILS_BACKUP}")

    original = UTILS_FILE.read_text(encoding="utf-8")
    patched = patch_text(original)

    if patched != original:
        UTILS_FILE.write_text(patched, encoding="utf-8")
        print(f"[OK] patched in place: {UTILS_FILE}")
    else:
        print(f"[OK] no changes needed: {UTILS_FILE}")


def copy_and_patch_analysis_scripts() -> None:
    for src, dst in FILES_TO_COPY.items():
        if not src.exists():
            print(f"[Skip] source not found: {src}")
            continue

        text = src.read_text(encoding="utf-8")
        patched = patch_text(text)

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(patched, encoding="utf-8")
        print(f"[OK] created: {dst}")


def main() -> None:
    copy_and_patch_analysis_scripts()
    patch_utils_model_utils()
    print("\nDone. You can now run the qwen25 analysis scripts in scripts/.")


if __name__ == "__main__":
    main()