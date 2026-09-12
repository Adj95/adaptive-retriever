"""Prompt manager loading externalized versioned prompt files from prompts/."""

from pathlib import Path
from typing import Tuple

class PromptLoader:
    def __init__(self, base_dir: Path | str | None = None):
        if base_dir is None:
            # Default to prompts/ directory relative to root
            self.base_dir = Path(__file__).resolve().parent.parent.parent / "prompts"
        else:
            self.base_dir = Path(base_dir)

    def load(self, prompt_path: str) -> Tuple[str, str]:
        """Loads prompt content and version header.
        
        Args:
            prompt_path: Relative path like 'transform/hyde' or 'transform/hyde.txt'
            
        Returns:
            Tuple of (prompt_text, version_string)
        """
        path = Path(prompt_path)
        if path.suffix != ".txt":
            path = path.with_suffix(".txt")

        full_path = self.base_dir / path
        if not full_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {full_path}")

        raw_text = full_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()

        if not lines or not lines[0].strip().startswith("# version:"):
            raise ValueError(f"Prompt file {full_path} missing mandatory '# version: X' header line.")

        version = lines[0].strip().split(":", 1)[1].strip()
        body_text = "\n".join(lines[1:]).strip()

        return body_text, version

# Global singleton loader
_prompt_loader = PromptLoader()

def load_prompt(prompt_path: str) -> Tuple[str, str]:
    return _prompt_loader.load(prompt_path)
