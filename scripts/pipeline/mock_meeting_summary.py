"""Generate a mock meeting summary for the Electron prototype."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class MockSummary:
    summary: str
    highlights: List[str]
    actions: List[str]
    attendees: List[str]
    duration_minutes: int
    generated_at: str


def generate_mock_summary(query: str, folder: str) -> MockSummary:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    base_summary = (
        "이 회의에서는 제품 출시 일정, 마케팅 전략, 팀 리소스 배분에 대해 논의했습니다. "
        "주요 의사결정으로는 출시일 확정과 다음 분기 예산 재조정이 포함돼 있습니다."
    )
    return MockSummary(
        summary=f"[모의 요약] '{query}' 요청에 대한 3분 요약입니다.\n{base_summary}",
        highlights=[
            "출시일을 2024년 11월 15일로 확정",
            "마케팅 예산 15% 증액 (디지털 채널 집중)",
            "후속 액션: 파트너사와 계약 조건 재협상",
        ],
        actions=[
            "PM팀: 10월 5일까지 베타 피드백 반영 계획 공유",
            "마케팅팀: SNS 캠페인 안 초안 10월 1일까지 제출",
            "CS팀: 주요 FAQ 업데이트 9월 말 완료",
        ],
        attendees=["김지훈", "박소영", "David Park", "이수진"],
        duration_minutes=45,
        generated_at=now,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="User query or meeting description")
    parser.add_argument("--folder", default="/data/meetings/mock", help="Folder context")
    parser.add_argument("--json", action="store_true", help="Return JSON output")
    args = parser.parse_args()

    summary = generate_mock_summary(args.query, args.folder)
    data = asdict(summary)
    data.update({"folder": args.folder, "query": args.query})

    if args.json:
        json.dump(data, fp=sys.stdout)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    main()
