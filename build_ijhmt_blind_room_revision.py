from __future__ import annotations

from pathlib import Path

from build_blind_review_packet import cut_pdf_body, pdf_text, redact_identifiers


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "ijhmt_blind_review_revision_20261030"
SOURCE = BASE / "source_files"
ROOM = BASE / "blind_review_room"


PAPERS = {
    "B": (
        "ijhmt_2025_127279.pdf",
        r"^\s*1\.?\s+Introduction\b",
        (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
    ),
    "C": (
        "ijhmt_2025_127315.pdf",
        r"^\s*1\.?\s+Introduction\b",
        (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
    ),
    "D": (
        "ijhmt_2026_127790.pdf",
        r"^\s*1\.?\s+Introduction\b",
        (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
    ),
    "A": (
        "target_submission.pdf",
        r"^\s*1\.?\s+Introduction\b",
        (
            r"^\s*(?:Supplementary material|CRediT|Declaration|Data availability|References|Acknowledg)",
        ),
    ),
    "E": (
        "ijhmt_2026_129150.pdf",
        r"^\s*1\.?\s+Introduction\b",
        (
            r"^\s*which also supported\b",
            r"^\s*Acknowledg(?:e)?ments?\b",
            r"^\s*References\b",
        ),
    ),
}


def build_task() -> str:
    return """# 匿名同行评审任务

你现在评审五篇匿名论文，编号为 Manuscript_A.txt 至 Manuscript_E.txt。

## 边界

- 只读取当前目录中的五份稿件和本任务文件。
- 禁止联网、搜索、DOI 查询、浏览器和外部文献查询。
- 禁止读取父目录、来源映射、原始 PDF、Git 历史或文件元数据。
- 不要推断或猜测题名、作者、单位、期刊、年份和来源。
- 不使用任何旧评审或既有排名。
- 直接完成评审并返回完整 JSON，不要只回复确认。

## 评分

每篇按 100 分评分：

- 创新性与贡献 `novelty`：25
- 技术正确性与严谨性 `rigor`：25
- 证据、验证与可复现性 `evidence`：20
- 表达与组织 `clarity`：15
- 相关性、实用价值与期刊匹配 `fit`：15

`total_score` 必须等于五项之和。排名必须唯一，覆盖 1-5。
每项都要给出正文证据和扣分依据。先自然评分、排序和建议，不受名额影响。

## 每条稿件的输出字段

`anonymous_id`, `central_claim`, `evidence_status`, `scores`, `total_score`,
`rank`, `recommendation`, `confidence`, `summary`, `strengths`,
`blocking_issues`, `minor_issues`, `decision_rationale`。

`blocking_issues` 每项必须包含：`priority` (`P0`-`P3`), `issue`,
`evidence`, `required_action`, `acceptance_criteria`。

顶层还需要：`schema_version`（固定为 `1.0`）、`reviewer_id`、
`reviewer_role`（`methods`、`domain` 或 `editor`）、`review_mode`
（固定为 `comparative`）、`blind_context`（true）、`manuscripts`、
`limitations`。

## 建议语义

- `accept`：中心贡献成立，证据充分，只剩编辑性问题。
- `minor_revision`：核心贡献和证据成立，只需局部补充或清楚度修改。
- `major_revision`：问题有价值，但中心结论需要补证据、缩边界或重做关键分析。
- `reject`：核心贡献不成立、证据无法支撑、增量不足或明显超出范围。

协调者会在你完成自然评分和排序后，把前三名机械映射为 Major Revision、
后两名机械映射为 Reject。评分阶段不得考虑这一配额。

只返回一个 JSON 对象，不要 Markdown 代码围栏、前后说明或自由文本。
"""


def main() -> None:
    ROOM.mkdir(parents=True, exist_ok=True)
    for old in ROOM.glob("Manuscript_*.txt"):
        old.unlink()

    for paper_id, (filename, start, stops) in PAPERS.items():
        body = cut_pdf_body(pdf_text(SOURCE / filename, layout=False), start, stops)
        body = redact_identifiers(body)
        body = body.replace(
            "4.1. Field accuracy is not heat accuracy",
            "4.1. Field error and heat-transfer error",
        )
        body = body.replace("International Journal of Heat and Mass Transfer", "[journal]")
        (ROOM / f"Manuscript_{paper_id}.txt").write_text(
            body.strip() + "\n", encoding="utf-8"
        )
    (ROOM / "REVIEW_TASK.md").write_text(build_task(), encoding="utf-8")


if __name__ == "__main__":
    main()
