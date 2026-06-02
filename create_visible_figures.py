from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "data" / "create_visible_figures.py"),
        run_name="__main__",
    )
