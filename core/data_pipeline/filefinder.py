# filefinder.py  (Step1: 파일 스캐너)
import os
import sys
import platform
import time
import threading
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any, Set, Union

try:
    import pwd  # POSIX only; optional
except ImportError:  # pragma: no cover - platform specific
    pwd = None

class StartupSpinner:
    """아주 이른 단계부터 '살아있음'을 보여주는 콘솔 스피너."""
    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, prefix: str = "", interval: float = 0.15):
        self.prefix = prefix
        self.interval = interval
        self._stop = threading.Event()
        self._t = None
        self._i = 0

    def start(self) -> None:
        if self._t:
            return

        def _run() -> None:
            while not self._stop.wait(self.interval):
                frame = self.FRAMES[self._i % len(self.FRAMES)]
                self._i += 1
                sys.stdout.write(f"\r{self.prefix} {frame} ")
                sys.stdout.flush()

        self._t = threading.Thread(target=_run, daemon=True)
        self._t.start()

    def stop(self, clear_line: bool = True) -> None:
        if not self._t:
            return
        self._stop.set()
        self._t.join()
        if clear_line:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

class FileFinder:
    DEFAULT_EXTS = {
        ".hwp", ".doc", ".docx",
        ".xlsx", ".xls", ".xlsm", ".xlsb", ".xltx",
        ".pdf",
        ".ppt", ".pptx",
        ".csv", ".txt", ".md", ".rst", ".json", ".py",
    }
    WINDOWS_SKIP_DIRS = {
        r"\$Recycle.Bin",
        r"\System Volume Information",
        r"\Windows\WinSxS\Temp",
        r"\Windows\Temp",
    }
    COMMON_SKIP_DIRS = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".git",
        ".svn",
        ".hg",
        ".idea",
        ".vscode",
        "node_modules",
        "bower_components",
        "vendor",
        "venv",
        ".venv",
        "env",
        ".env",
        "dist",
        "build",
        "tmp",
        "temp",
        ".cache",
        "logs",
        "applications",
        "system",
    }

    def __init__(
        self,
        exts: Optional[Iterable[str]] = None,
        scan_all_drives: bool = True,
        start_from_current_drive_only: bool = False,
        follow_symlinks: bool = False,
        max_depth: Optional[int] = None,
        show_progress: bool = True,
        progress_update_secs: float = 0.5,
        estimate_total_dirs: bool = False,
        startup_banner: bool = True,
        skip_dir_names: Optional[Iterable[str]] = None,
        skip_hidden: bool = True,
    ):
        self.exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or self.DEFAULT_EXTS)}
        self.scan_all_drives = scan_all_drives
        self.start_from_current_drive_only = start_from_current_drive_only
        self.follow_symlinks = follow_symlinks
        if platform.system().lower().startswith("win") and self.follow_symlinks:
            # 윈도우는 샌드박스/권한 제한으로 symlink 추적이 불필요하므로 강제로 비활성화
            self.follow_symlinks = False
            print("ℹ️ Windows 환경에서는 symlink 추적을 비활성화했습니다.", flush=True)
        self.max_depth = max_depth
        self.show_progress = show_progress
        self.progress_update_secs = progress_update_secs
        self.estimate_total_dirs = estimate_total_dirs
        self.startup_banner = startup_banner
        combined_skips = set(self.COMMON_SKIP_DIRS)
        if skip_dir_names:
            combined_skips.update(skip_dir_names)
        self.skip_hidden = skip_hidden
        self._skip_dir_names = {name.strip().lower() for name in combined_skips if name}

        self._lock = threading.Lock()
        self._dirs_scanned = 0
        self._files_scanned = 0
        self._matched = 0
        self._start_time = None
        self._stop_progress = threading.Event()
        self._total_dirs_estimate = None
        self._current_root = ""
        self._permission_notices: Set[str] = set()

    # ---------- roots ----------
    def _windows_drives(self) -> List[Path]:
        roots = []
        for code in range(ord('A'), ord('Z') + 1):
            root = Path(f"{chr(code)}:/")
            if root.exists():
                roots.append(root)
        return roots
    def _mac_roots(self) -> List[Path]:
        roots = [Path("/")]
        vol = Path("/Volumes")
        if vol.exists():
            roots.extend([p for p in vol.iterdir() if p.is_dir()])
        return roots
    def _linux_roots(self) -> List[Path]:
        roots = [Path("/")]
        for extra in ("/mnt", "/media"):
            p = Path(extra)
            if p.exists():
                roots.extend([c for c in p.iterdir() if c.is_dir()])
        return roots
    def get_roots(self) -> List[Path]:
        system = platform.system().lower()
        if system.startswith("win"):
            if self.start_from_current_drive_only:
                cwd = Path.cwd().resolve()
                return [Path(cwd.anchor)] if cwd.anchor else [Path("C:/")]
            return self._windows_drives()
        elif system == "darwin":
            return self._mac_roots()
        else:
            return self._linux_roots()

    # ---------- utils ----------
    def _should_skip_dir(self, path: Path) -> bool:
        if platform.system().lower().startswith("win"):
            path_str = str(path)
            for skip in self.WINDOWS_SKIP_DIRS:
                if skip.lower() in path_str.lower():
                    return True
        try:
            name = path.name
        except Exception:
            name = ""
        if name:
            normalized = name.strip().lower()
            if normalized in self._skip_dir_names:
                return True
            if self.skip_hidden and name.startswith('.') and path.parent != path:
                return True
        return False
    def _depth_from_root(self, root: Path, current: Path) -> int:
        try:
            rel = current.resolve().relative_to(root.resolve())
        except Exception:
            return 0
        if str(rel) == ".":
            return 0
        return len(rel.parts)

    # ---------- preflight ----------
    def preflight(self) -> List[Path]:
        roots = self.get_roots()
        print("🧭 루트/드라이브 감지 완료:", flush=True)
        for r in roots:
            can_read = True
            try:
                os.listdir(r)
            except Exception:
                can_read = False
            print(f"  • {str(r)}  (readable={'Y' if can_read else 'N'})", flush=True)
        if self.max_depth is not None:
            print(f"🔧 최대 깊이 제한: {self.max_depth}", flush=True)
        print("⏳ 준비 중… 잠시만요.", flush=True)
        return roots

    # ---------- progress ----------
    def _progress_loop(self):
        last_dirs = 0
        last_time = self._start_time
        while not self._stop_progress.wait(self.progress_update_secs):
            with self._lock:
                dirs = self._dirs_scanned
                files = self._files_scanned
                matched = self._matched
                root = self._current_root
                total_dirs = self._total_dirs_estimate

            now = time.time()
            dt = max(1e-6, now - last_time)
            dps = (dirs - last_dirs) / dt
            last_dirs, last_time = dirs, now
            elapsed = now - self._start_time

            if total_dirs and total_dirs > 0:
                pct = min(100.0, (dirs / total_dirs) * 100.0)
                remaining = max(0.0, total_dirs - dirs)
                eta = (remaining / dps) if dps > 0 else float('inf')
                pct_str = f"{pct:5.1f}%"
                eta_str = self._fmt_secs(eta)
            else:
                pct_str = "  N/A"
                eta_str = "--:--"

            line = (f"[{pct_str}] root={root} | dirs={dirs:,} | files={files:,} "
                    f"| found={matched:,} | {dps:,.1f} dir/s | elapsed={self._fmt_secs(elapsed)} | ETA={eta_str}   ")
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

    @staticmethod
    def _fmt_secs(s: float) -> str:
        if s == float("inf"):
            return "∞"
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    def _estimate_total_dirs(self, roots: List[Path]) -> int:
        total = 0
        print("📐 디렉토리 개수 추정 시작… (퍼센트 제공용)", flush=True)
        spinner = StartupSpinner(prefix="  sampling", interval=0.2)
        spinner.start()
        try:
            for root in roots:
                print(f"\n  └ 샘플링: {root}", flush=True)
                stack = [root]
                visited_inodes = set()
                while stack:
                    current = stack.pop()
                    if self.max_depth is not None and self._depth_from_root(root, current) > self.max_depth:
                        continue
                    try:
                        with os.scandir(current) as it:
                            for entry in it:
                                try:
                                    if entry.is_symlink() and not self.follow_symlinks:
                                        continue
                                    if entry.is_dir(follow_symlinks=self.follow_symlinks):
                                        p = Path(entry.path)
                                        if self._should_skip_dir(p):
                                            continue
                                        if hasattr(entry, 'inode'):
                                            key = (entry.inode(), entry.stat(follow_symlinks=False).st_dev)
                                            if key in visited_inodes:
                                                continue
                                            visited_inodes.add(key)
                                        total += 1
                                        if total % 500 == 0:
                                            sys.stdout.write(f"\r      누적 디렉토리: {total:,}")
                                            sys.stdout.flush()
                                        stack.append(p)
                                except (PermissionError, FileNotFoundError, OSError):
                                    continue
                    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
                        continue
        finally:
            spinner.stop()
            print(f"\n✅ 추정 완료: 총 디렉토리 ≈ {total:,}", flush=True)
        return total

    # ---------- dfs ----------
    def _iter_files(self, root: Path):
        stack = [root]
        visited_inodes = set()
        while stack:
            current = stack.pop()
            with self._lock:
                self._dirs_scanned += 1
            if self.max_depth is not None and self._depth_from_root(root, current) > self.max_depth:
                continue
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            if entry.is_symlink() and not self.follow_symlinks:
                                continue
                            if entry.is_dir(follow_symlinks=self.follow_symlinks):
                                p = Path(entry.path)
                                if self._should_skip_dir(p):
                                    continue
                                if hasattr(entry, 'inode'):
                                    key = (entry.inode(), entry.stat(follow_symlinks=False).st_dev)
                                    if key in visited_inodes:
                                        continue
                                    visited_inodes.add(key)
                                stack.append(p)
                                continue
                            if entry.is_file(follow_symlinks=self.follow_symlinks):
                                with self._lock:
                                    self._files_scanned += 1
                                yield Path(entry.path)
                        except PermissionError:
                            self._record_permission(entry.path)
                            continue
                        except (FileNotFoundError, OSError):
                            continue
            except PermissionError:
                self._record_permission(current)
                continue
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue

    # ---------- public ----------
    def find(self, roots: Optional[List[Path]] = None, run_async: bool = False) -> List[Dict]:
        self._start_time = time.time()
        self._dirs_scanned = self._files_scanned = self._matched = 0
        self._stop_progress.clear()

        spinner = None
        if self.startup_banner:
            spinner = StartupSpinner(prefix="🔎 초기화", interval=0.1)
            spinner.start()
        try:
            roots = roots or self.preflight()
        finally:
            if spinner:
                spinner.stop()

        if self.estimate_total_dirs:
            self._total_dirs_estimate = self._estimate_total_dirs(roots)
        else:
            self._total_dirs_estimate = None

        progress_thread = None
        if self.show_progress:
            progress_thread = threading.Thread(target=self._progress_loop, daemon=True)
            progress_thread.start()

        try:
            results: List[Dict] = []
            def _scan():
                for root in roots:
                    with self._lock:
                        self._current_root = str(root)
                    for f in self._iter_files(root):
                        ext = f.suffix.lower()
                        if ext in self.exts:
                            try:
                                st = f.stat()
                                owner = self._resolve_owner(st)
                                results.append({
                                    "path": str(f),
                                    "size": st.st_size,
                                    "mtime": st.st_mtime,
                                    "ctime": st.st_ctime,
                                    "ext": ext,
                                    "drive": f.anchor,
                                    "owner": owner,
                                })
                                with self._lock:
                                    self._matched += 1
                            except (FileNotFoundError, PermissionError, OSError):
                                continue
            if run_async:
                worker = threading.Thread(target=_scan, daemon=True)
                worker.start()
                worker.join()
            else:
                _scan()
            results.sort(key=lambda x: x["mtime"], reverse=True)
            return results
        finally:
            self._stop_progress.set()
            if progress_thread:
                progress_thread.join()

    @staticmethod
    def human_time(epoch_sec: float) -> str:
        t = time.localtime(epoch_sec)
        return time.strftime("%Y-%m-%d %H:%M:%S", t)

    @staticmethod
    def to_csv(rows: List[Dict], out_path: Path) -> None:
        import csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["path", "size", "mtime", "ctime", "ext", "drive", "owner"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _record_permission(self, path: Union[Path, str]) -> None:
        key = str(path)
        if key not in self._permission_notices:
            self._permission_notices.add(key)
            if len(self._permission_notices) <= 20:
                print(f"⚠️ 접근 권한이 없어 건너뜁니다: {key}", flush=True)

    @staticmethod
    def collect_file_metadata(path: Path, *, allowed_exts: Optional[Iterable[str]] = None) -> Optional[Dict[str, Any]]:
        """Build a single metadata row for `path` using the scanner's schema."""
        try:
            p = Path(path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None

        ext = p.suffix.lower()
        if allowed_exts is not None:
            normalized = {
                (e.lower() if e.startswith(".") else f".{e.lower()}")
                for e in allowed_exts
            }
            if ext not in normalized:
                return None

        try:
            st = p.stat()
        except (FileNotFoundError, PermissionError, OSError):
            return None

        owner = FileFinder._resolve_owner(st)
        return {
            "path": str(p),
            "size": st.st_size,
            "mtime": st.st_mtime,
            "ctime": st.st_ctime,
            "ext": ext,
            "drive": p.anchor,
            "owner": owner,
        }

    @staticmethod
    def _resolve_owner(stat_result) -> str:
        if not stat_result:
            return ""
        if pwd is not None:
            try:
                return pwd.getpwuid(stat_result.st_uid).pw_name  # type: ignore[attr-defined]
            except (KeyError, AttributeError):
                return ""
        return ""
