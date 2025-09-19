# infopilot.py
from pathlib import Path
import argparse

# 모듈 임포트
from encoding_utils import detect_file_encodings
from filefinder import FileFinder
def cmd_scan(args):
    include_paths = [Path(p).expanduser().resolve() for p in args.path]
    for p in include_paths:
        if not p.exists():
            raise FileNotFoundError(f"지정된 경로를 찾을 수 없습니다: {p}")
    finder = FileFinder(
        exts=FileFinder.DEFAULT_EXTS,
        scan_all_drives=False,
        start_from_current_drive_only=True,
        follow_symlinks=False,
        max_depth=None,
        show_progress=True,
        progress_update_secs=0.5,
        estimate_total_dirs=False,
        startup_banner=True,
        include_paths=include_paths,
        exclude_tokens=args.exclude,
    )
    print("🔍 지원 확장자:", ", ".join(sorted(finder.exts)))
    files = finder.find(roots=include_paths, run_async=False)
    out = Path(args.out)
    FileFinder.to_csv(files, out)
    print(f"📦 스캔 결과 저장: {out}")


def cmd_train(args):
    from pipeline import run_step2, TrainConfig
    import csv
    rows = []
    scan_csv_path = getattr(args, "input", None) or args.scan_csv
    model_path_override = getattr(args, "index", None)
    scan_path = Path(scan_csv_path)
    last_error = None
    used_encoding = None
    for enc in detect_file_encodings(scan_path):
        try:
            with scan_path.open("r", encoding=enc, newline="") as f:
                for r in csv.DictReader(f):
                    r["size"] = int(r.get("size") or 0)
                    r["mtime"] = float(r.get("mtime") or 0.0)
                    rows.append(r)
            used_encoding = enc
            break
        except UnicodeDecodeError as e:
            last_error = e
            rows.clear()
            continue
        except FileNotFoundError:
            raise
        except Exception as e:
            last_error = e
            rows.clear()
            continue
    if used_encoding is None:
        raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "scan csv encoding detection failed")
    print(f"🗂️ 스캔 CSV 인코딩: {used_encoding}")

    cfg = TrainConfig(
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    out_corpus = Path(args.corpus)
    out_model = Path(model_path_override or args.model)
    # run_step2 호출 시 translate 플래그 전달
    df, tm = run_step2(rows, out_corpus=out_corpus, out_model=out_model, cfg=cfg, use_tqdm=True, translate=args.translate)
    print("✅ 학습 완료")


def cmd_chat(args):
    """대화형 검색 모드 (LNPChat 사용)"""
    from lnp_chat import LNPChat # 새로운 LNP Chat 클래스를 임포트
    # LNPChat 클래스 인스턴스 생성 및 준비
    chat_session = LNPChat(
        model_path=Path(args.model),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        topk=args.topk,
        translate=args.translate # 번역 옵션 전달
    )
    chat_session.ready(rebuild=getattr(args, "rebuild", False))

    print("\n💬 InfoPilot Chat 모드입니다. (종료하려면 'exit' 또는 '종료' 입력)")
    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "종료"}:
            print("👋 종료합니다.")
            break
        
        # LNPChat의 ask 메소드 호출
        result = chat_session.ask(query)
        print(result["answer"])
        if result.get("suggestions"):
            print("\n💡 이런 질문은 어떠세요?")
            for s in result["suggestions"]:
                print(f"   - {s}")
        print("-" * 80)


def cmd_pipeline(args):
    scan_args = argparse.Namespace(
        path=args.path,
        exclude=args.exclude,
        out=args.scan_out,
    )
    cmd_scan(scan_args)

    train_args = argparse.Namespace(
        scan_csv=args.scan_out,
        input=None,
        corpus=args.corpus,
        model=args.model,
        index=None,
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        min_df=args.min_df,
        max_df=args.max_df,
        translate=args.translate,
    )
    cmd_train(train_args)

    chat_args = argparse.Namespace(
        model=args.model,
        corpus=args.corpus,
        cache=args.cache,
        topk=args.topk,
        translate=args.translate,
        rebuild=True,
    )
    cmd_chat(chat_args)


def main():
    ap = argparse.ArgumentParser(prog="infopilot", description="InfoPilot CLI - 다국어 문서 검색기")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # scan
    ap_scan = sp.add_parser(
        "scan",
        help="드라이브 스캔하여 파일 목록 수집 (기본: HWP, DOC/DOCX, XLSX/XLS, PDF, PPT/PPTX, CSV, TXT)"
    )
    ap_scan.add_argument(
        "--path",
        required=True,
        nargs="+",
        help="스캔할 루트 폴더 경로 (여러 개 지정 가능)",
    )
    ap_scan.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="스캔에서 제외할 디렉터리 이름/패턴 (여러 번 사용 가능)",
    )
    ap_scan.add_argument("--out", default="./data/found_files.csv")
    ap_scan.set_defaults(func=cmd_scan)

    # train
    ap_train = sp.add_parser("train", help="코퍼스 생성 + 모델 학습 (기본: 번역 활성)")
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--input", help="scan CSV 경로 (구 --scan_csv 별칭)")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--model", default="./data/topic_model.joblib")
    ap_train.add_argument("--index", help="모델 파일 경로 (구 --model 별칭)")
    ap_train.add_argument("--max_features", type=int, default=150000)
    ap_train.add_argument("--n_components", type=int, default=128)
    ap_train.add_argument("--n_clusters", type=int, default=25)
    ap_train.add_argument("--min_df", type=int, default=2)
    ap_train.add_argument("--max_df", type=float, default=0.85)
    ap_train.add_argument("--no-translate", dest="translate", action="store_false", help="번역 기능을 비활성화하고 원문으로 학습합니다.")
    ap_train.set_defaults(translate=True)
    ap_train.set_defaults(func=cmd_train)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 질의 모드 (기본: 번역 활성)")
    ap_chat.add_argument("--model", default="./data/topic_model.joblib")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--topk", type=int, default=5)
    ap_chat.add_argument("--rebuild", action="store_true", help="기존 인덱스를 무시하고 재생성합니다.")
    ap_chat.add_argument("--no-translate", dest="translate", action="store_false", help="질문 번역 기능을 비활성화합니다.")
    ap_chat.set_defaults(translate=True)
    ap_chat.set_defaults(func=cmd_chat)

    # pipeline
    ap_pipeline = sp.add_parser("pipeline", help="scan→train→chat 원스텝 실행")
    ap_pipeline.add_argument("--path", required=True, nargs="+", help="스캔할 루트 폴더 경로")
    ap_pipeline.add_argument("--exclude", action="append", default=[], help="스캔에서 제외할 디렉터리/패턴")
    ap_pipeline.add_argument("--scan-out", dest="scan_out", default="./data/found_files.csv")
    ap_pipeline.add_argument("--corpus", default="./data/corpus.parquet")
    ap_pipeline.add_argument("--model", default="./data/topic_model.joblib")
    ap_pipeline.add_argument("--cache", default="./index_cache")
    ap_pipeline.add_argument("--topk", type=int, default=5)
    ap_pipeline.add_argument("--max_features", type=int, default=150000)
    ap_pipeline.add_argument("--n_components", type=int, default=128)
    ap_pipeline.add_argument("--n_clusters", type=int, default=25)
    ap_pipeline.add_argument("--min_df", type=int, default=2)
    ap_pipeline.add_argument("--max_df", type=float, default=0.85)
    ap_pipeline.add_argument("--no-translate", dest="translate", action="store_false", help="번역 기능을 비활성화합니다.")
    ap_pipeline.set_defaults(translate=True)
    ap_pipeline.set_defaults(func=cmd_pipeline)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
