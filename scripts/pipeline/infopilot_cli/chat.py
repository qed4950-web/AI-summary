from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

from core.agents.document import DocumentAgent, DocumentAgentConfig
from core.agents.meeting import MeetingAgent
from core.agents.photo import PhotoAgent
from core.data_pipeline.pipeline import run_step2
from core.data_pipeline.policies.engine import PolicyEngine
from core.errors import PolicyViolationError

from core.conversation.orchestrator import AssistantOrchestrator

from .history import load_agent_history, remember_agent_history
from .policy import enforce_cache_limit, load_policy_engine
from .scan_rows import load_scan_rows, resolve_scan_csv
from .train_config import default_train_config


def ensure_chat_artifacts(
    *,
    scan_csv: Path,
    corpus: Path,
    model: Path,
    translate: bool,
    auto_train: bool,
    policy_engine: Optional[PolicyEngine],
    policy_agent: str,
) -> bool:
    """Ensure chat artifacts exist and are up to date. Returns True if training ran."""

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    resolved_scan: Optional[Path] = None
    if scan_csv:
        try:
            resolved_scan = resolve_scan_csv(scan_csv)
        except FileNotFoundError:
            resolved_scan = None

    artifacts_exist = corpus.exists() and model.exists()
    needs_train = not artifacts_exist

    if not needs_train and resolved_scan:
        scan_mtime = mtime(resolved_scan)
        artifacts_mtime = min(mtime(corpus), mtime(model))
        if scan_mtime > artifacts_mtime:
            needs_train = True

    if not needs_train:
        print("🔄 인덱스 최신성 확인 완료.")
        return False

    if resolved_scan is None:
        msg = (
            "⚠️ 학습 산출물이 없거나 오래되었지만 사용할 스캔 CSV를 찾지 못했습니다."
            " `--scan_csv` 경로를 확인하거나 'scan' 명령을 다시 실행해주세요."
        )
        raise FileNotFoundError(msg)

    if not auto_train:
        raise RuntimeError(
            "학습 산출물이 최신이 아닙니다. 'python infopilot.py train --scan_csv "
            f"{resolved_scan}'를 실행한 뒤 다시 시도해주세요."
        )

    print("⚠️ 스캔 결과가 모델보다 최신입니다. 자동으로 train 단계를 실행합니다.")
    rows = list(
        load_scan_rows(
            resolved_scan,
            policy_engine=policy_engine,
            include_manual=False,
            agent=policy_agent,
        )
    )
    if not rows:
        raise ValueError("자동 학습을 위한 유효한 행이 없습니다. 스캔 결과를 확인해주세요.")

    cfg = default_train_config()
    run_step2(
        rows,
        out_corpus=corpus,
        out_model=model,
        cfg=cfg,
        use_tqdm=True,
        translate=translate,
    )
    print("✅ 자동 학습 완료")
    return True


def cmd_chat(
    args,
    *,
    default_policy_path: Path,
    policy_agent: str,
) -> None:
    policy_arg = getattr(args, "policy", None)
    policy_normalized = (policy_arg or "").strip().lower()
    policy_required = policy_normalized != "none"
    policy_engine = load_policy_engine(
        policy_arg,
        default_policy_path=default_policy_path,
        fail_if_missing=policy_required,
        stage="chat",
    )

    def _env_or_arg(name: str, default: Optional[str] = None) -> Optional[str]:
        value = getattr(args, name, None)
        if value:
            value = str(value).strip()
            if value:
                return value
        env_name = f"LNPCHAT_{name.upper()}"
        env_value = os.getenv(env_name)
        if env_value is None:
            return default
        env_value = env_value.strip()
        return env_value or default

    llm_backend = _env_or_arg("llm_backend")
    llm_model = _env_or_arg("llm_model", default="llama3")
    llm_host = _env_or_arg("llm_host", default="")

    auto_trained = ensure_chat_artifacts(
        scan_csv=Path(args.scan_csv),
        corpus=Path(args.corpus),
        model=Path(args.model),
        translate=args.translate,
        auto_train=args.auto_train,
        policy_engine=policy_engine,
        policy_agent=policy_agent,
    )

    document_agent = DocumentAgent(
        DocumentAgentConfig(
            model_path=Path(args.model),
            corpus_path=Path(args.corpus),
            cache_dir=Path(args.cache),
            topk=args.topk,
            translate=args.translate,
            rerank=args.rerank,
            rerank_model=args.rerank_model,
            rerank_depth=args.rerank_depth,
            rerank_batch_size=args.rerank_batch_size,
            rerank_device=args.rerank_device or None,
            rerank_min_score=args.rerank_min_score,
            lexical_weight=args.lexical_weight,
            show_translation=args.show_translation,
            translation_lang=args.translation_lang,
            min_similarity=args.min_similarity,
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_host=llm_host,
            llm_options={},
            policy_engine=policy_engine if policy_engine and policy_engine.has_policies else policy_engine,
            policy_scope=(getattr(args, "scope", "auto") or "auto").lower(),
            policy_agent=policy_agent,
            rebuild_index=auto_trained,
        )
    )
    meeting_agent = MeetingAgent()
    photo_agent = PhotoAgent()

    orchestrator = AssistantOrchestrator(
        [document_agent, meeting_agent, photo_agent],
        llm_client=document_agent.llm_client,
    )

    def _print_response(resp) -> None:
        prefix = f"[{resp.agent}] " if getattr(resp, "agent", None) else ""
        print(prefix + getattr(resp, "message", ""))
        suggestions = getattr(resp, "suggestions", None) or []
        if suggestions:
            print("\n💡 이런 질문은 어떠세요?")
            for suggestion in suggestions:
                print(f"   - {suggestion}")

    def _cli_progress_handler(agent_label: str) -> Callable[[Dict[str, Any]], None]:
        def _handler(event: Dict[str, Any]) -> None:
            stage = event.get("stage")
            status = event.get("status")
            prefix = f"[{agent_label}]"
            if status == "running":
                print(f"{prefix} ▶ {stage} 시작")
            elif status == "completed":
                print(f"{prefix} ✅ {stage} 완료")
            elif status == "failed":
                error = event.get("error")
                print(f"{prefix} ❌ {stage} 실패: {error}")
            elif status == "cancelled":
                print(f"{prefix} ⛔ {stage} 취소")
        return _handler

    def _prompt_follow_up(reason: Optional[str], message: str) -> Optional[Dict[str, object]]:
        print(message)
        history = load_agent_history()
        if reason == "needs_audio":
            recent = history.get("meeting_audio", [])
            if recent:
                print("\n📁 최근 사용한 오디오 파일:")
                for idx, item in enumerate(recent, start=1):
                    print(f"  {idx}. {item}")
            prompt = "회의 요약을 실행하려면 오디오 파일 전체 경로를 입력하거나 번호를 선택하세요> "
            raw = input(prompt).strip()
            if not raw:
                print("⚠️ 경로를 입력하지 않아 요청을 취소했습니다.")
                return None
            if raw.isdigit():
                index = int(raw) - 1
                if 0 <= index < len(recent):
                    audio_path = recent[index]
                else:
                    print("⚠️ 번호가 유효하지 않아 요청을 취소했습니다.")
                    return None
            else:
                audio_path = raw
            remember_agent_history("meeting_audio", [audio_path])
            return {"audio_path": audio_path, "enable_resume": True}
        if reason == "needs_roots":
            recent = history.get("photo_roots", [])
            if recent:
                print("\n📸 최근 사용한 사진 폴더:")
                for idx, item in enumerate(recent, start=1):
                    print(f"  {idx}. {item}")
            prompt = "사진 폴더 경로를 입력하거나 번호(여러 개는 콤마)로 선택하세요> "
            raw = input(prompt).strip()
            if not raw:
                print("⚠️ 경로를 입력하지 않아 요청을 취소했습니다.")
                return None
            roots: List[str] = []
            for token in [part.strip() for part in raw.split(",") if part.strip()]:
                if token.isdigit():
                    index = int(token) - 1
                    if 0 <= index < len(recent):
                        roots.append(recent[index])
                    else:
                        print(f"⚠️ 번호 {token}가 유효하지 않아 무시합니다.")
                else:
                    roots.append(token)
            if not roots:
                print("⚠️ 유효한 경로가 없어 요청을 취소했습니다.")
                return None
            remember_agent_history("photo_roots", roots)
            return {"roots": roots}
        extra = input("추가 정보를 입력하세요> ").strip()
        if not extra:
            print("⚠️ 추가 정보를 입력하지 않아 요청을 취소했습니다.")
            return None
        return {"details": extra}

    def _resolve_follow_up(original_query: str, initial_response) -> Any:
        response = initial_response
        while getattr(response, "agent", None) == "follow_up":
            follow_context = _prompt_follow_up(getattr(response, "reason", None), getattr(response, "message", ""))
            if not follow_context:
                break
            if getattr(response, "reason", None) == "needs_audio":
                follow_context.setdefault("__progress_callback", _cli_progress_handler("회의 비서"))
            elif getattr(response, "reason", None) == "needs_roots":
                follow_context.setdefault("__progress_callback", _cli_progress_handler("사진 비서"))
            response = orchestrator.handle(original_query, follow_context)
        return response

    single_query = getattr(args, "query", None)
    json_mode = bool(getattr(args, "json", False))
    if json_mode and not single_query:
        raise SystemExit("--json 옵션은 --query와 함께 사용해야 합니다.")

    enforce_cache_limit(
        Path(args.cache),
        policy_engine,
        hard_limit=getattr(args, "cache_hard_limit", False),
        clean_on_limit=getattr(args, "cache_clean_on_limit", False),
    )

    if single_query:
        response = orchestrator.handle(single_query)
        if getattr(response, "agent", None) == "follow_up" and not json_mode:
            response = _resolve_follow_up(single_query, response)
        if json_mode:
            metadata = response.metadata if isinstance(response.metadata, dict) else {}
            payload: Dict[str, Any] = {
                "query": single_query,
                "answer": response.message,
                "agent": response.agent,
                "reason": response.reason,
                "metadata": metadata,
                "suggestions": response.suggestions or [],
                "results": [],
            }
            if response.agent == "follow_up":
                metadata["follow_up"] = response.reason
            hits = response.metadata.get("hits", []) if isinstance(response.metadata, dict) else []
            for hit in hits[: args.topk]:
                payload["results"].append(
                    {
                        "title": Path(str(hit.get("path") or "")).name,
                        "path": hit.get("path"),
                        "ext": hit.get("ext"),
                        "score": hit.get("similarity", hit.get("vector_similarity")),
                        "vector_score": hit.get("vector_similarity"),
                        "lexical_score": hit.get("lexical_score"),
                        "match_reasons": hit.get("match_reasons") or [],
                        "preview": hit.get("preview"),
                        "translation": hit.get("translation"),
                    }
                )
            print(json.dumps(payload, ensure_ascii=False))
        else:
            _print_response(response)
        return

    print(
        "\n💬 InfoPilot Chat 모드입니다. 자유롭게 대화하고, 문서 검색이 필요하면 '/search 질문'처럼 입력해 보세요. "
        "(종료하려면 'exit' 또는 '종료' 입력)"
    )
    print("   명령어: /search <질문>, /meeting, /photo")
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

        response = orchestrator.handle(query)
        response = _resolve_follow_up(query, response)
        _print_response(response)
        print("-" * 80)


__all__ = ["cmd_chat", "ensure_chat_artifacts"]
