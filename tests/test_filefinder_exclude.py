import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filefinder import FileFinder

def make_file(path: Path, content: str = "sample") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8-sig")
    return path


def test_default_excludes_skip_virtual_env(tmp_path):
    keep_dir = tmp_path / "docs"
    skip_dir = tmp_path / ".venv" / "lib"
    keep_file = make_file(keep_dir / "keep.txt", "keep me")
    make_file(skip_dir / "ignore.txt", "skip me")

    finder = FileFinder(
        include_paths=[tmp_path],
        scan_all_drives=False,
        start_from_current_drive_only=True,
        follow_symlinks=False,
        show_progress=False,
        startup_banner=False,
    )

    results = finder.find(roots=[tmp_path], run_async=False)
    paths = {Path(r["path"]) for r in results}

    assert keep_file in paths
    assert all(".venv" not in Path(r["path"]).parts for r in results)


def test_symlink_outside_root_is_ignored(tmp_path):
    outside = tmp_path.parent / "outside-root"
    outside.mkdir(parents=True, exist_ok=True)
    outside_file = make_file(outside / "secret.txt", "secret")

    link_dir = tmp_path / "linked"
    os.symlink(outside, link_dir, target_is_directory=True)

    finder = FileFinder(
        include_paths=[tmp_path],
        scan_all_drives=False,
        start_from_current_drive_only=True,
        follow_symlinks=True,
        show_progress=False,
        startup_banner=False,
    )

    results = finder.find(roots=[tmp_path], run_async=False)
    paths = {Path(r["path"]).resolve() for r in results}

    assert outside_file.resolve() not in paths
