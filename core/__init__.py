import pathlib

if (pathlib.Path(__file__).parent / "main_description.md").exists():
    __doc__ = (pathlib.Path(__file__).parent / "main_description.md").read_text(encoding="utf-8", errors="ignore")
else:
    __doc__ = "⚠️ Documentation file not found: `main_description.md`"
