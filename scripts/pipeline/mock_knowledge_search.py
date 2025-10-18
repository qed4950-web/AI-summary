"""Generate mock knowledge search results for the Electron prototype."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List


@dataclass
class MockDocument:
    title: str
    snippet: str
    path: str


@dataclass
class MockSearchPayload:
    query: str
    folder: str
    generated_at: str
    items: List[MockDocument]


def generate_mock_results(query: str, folder: str) -> MockSearchPayload:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    templates = [
        MockDocument(
            title="AI 전략 로드맵",
            snippet="AI 거버넌스, 데이터 인프라, 서비스 발굴 로드맵 요약.",
            path=f"{folder.rstrip('/')}/ai_strategy_roadmap.docx",
        ),
        MockDocument(
            title="{keyword} 관련 회의 메모",
            snippet="핵심 의사결정과 후속 액션이 정리된 문서.",
            path=f"{folder.rstrip('/')}/meeting_notes_{now.replace('-', '').replace(':', '')}.md",
        ),
        MockDocument(
            title="{keyword} 요약본",
            snippet="검색 키워드와 연관된 요약/보고 템플릿.",
            path=f"{folder.rstrip('/')}/summary_{now.replace('-', '').replace(':', '')}.txt",
        ),
    ]

    keyword = query.split()[0] if query.split() else "문서"
    items = [
        MockDocument(
            title=t.title.format(keyword=keyword),
            snippet=t.snippet,
            path=t.path,
        )
        for t in templates
    ]

    return MockSearchPayload(query=query, folder=folder, generated_at=now, items=items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--folder", default="/data/plans", help="Smart folder path")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = generate_mock_results(args.query, args.folder)
    data = asdict(payload)

    if args.json:
        json.dump(data, fp=sys.stdout, ensure_ascii=False)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
