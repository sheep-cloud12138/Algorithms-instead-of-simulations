from __future__ import annotations

from pathlib import Path

from build_blind_review_packet import cut_pdf_body, pdf_text, redact_identifiers


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "ijhmt_blind_review_20261029"
SOURCE = BASE / "source_files"
ROOM = BASE / "blind_review_room"


def main() -> None:
    ROOM.mkdir(parents=True, exist_ok=True)
    for old in ROOM.glob("Manuscript_*.txt"):
        old.unlink()

    papers = {
        "A": (
            SOURCE / "ijhmt_2025_127279.pdf",
            r"^\s*1\.?\s+Introduction\b",
            (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
        ),
        "B": (
            SOURCE / "ijhmt_2025_127315.pdf",
            r"^\s*1\.?\s+Introduction\b",
            (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
        ),
        "C": (
            SOURCE / "ijhmt_2026_127790.pdf",
            r"^\s*1\.?\s+Introduction\b",
            (r"^\s*Acknowledg(?:e)?ments?\b", r"^\s*References\b"),
        ),
        "D": (
            SOURCE / "target_submission.pdf",
            r"^\s*1\.?\s+Introduction\b",
            (
                r"^\s*(?:Supplementary material|CRediT|Declaration|Data availability|References|Acknowledg)",
            ),
        ),
        "E": (
            SOURCE / "ijhmt_2026_129150.pdf",
            r"^\s*1\.?\s+Introduction\b",
            (
                r"^\s*which also supported\b",
                r"^\s*Acknowledg(?:e)?ments?\b",
                r"^\s*References\b",
            ),
        ),
    }

    for paper_id, (path, start, stops) in papers.items():
        body = cut_pdf_body(pdf_text(path, layout=False), start, stops)
        body = redact_identifiers(body)
        body = body.replace(
            "4.1. Field accuracy is not heat accuracy",
            "4.1. Field error and heat-transfer error",
        )
        (ROOM / f"Manuscript_{paper_id}.txt").write_text(
            body.strip() + "\n",
            encoding="utf-8",
        )

    task = """# 匿名同行评审任务

你现在评审五篇匿名论文，编号为 Manuscript_A.txt 至 Manuscript_E.txt。

## 边界

- 只读取当前目录中的五份稿件和本任务文件。
- 禁止联网、搜索、DOI 查询、浏览器和外部文献查询。
- 禁止读取父目录、来源映射、原始 PDF、Git 历史或文件元数据。
- 不要推断或猜测题名、作者、单位、期刊、年份和来源。
- 直接完成评审并返回完整报告，不要只回复确认。

## 评分

每篇按 100 分评分：

- 创新性与贡献：25
- 技术正确性与严谨性：25
- 证据、验证与可复现性：20
- 表达与组织：15
- 相关性、实用价值与期刊匹配：15

必须说明每一项的扣分依据，并给出正文证据。

## 输出

1. 五篇排名表：排名、匿名编号、分项分数、总分、建议。
2. 每篇一句话概括、主要优点、决定性/重大问题、最低修改要求、次要问题。
3. 横向比较：前两名的区别、末两名为何落后。
4. 最终建议：Accept / Minor Revision / Major Revision / Reject。

评分时不要考虑任何名额。全部评分完成后再给建议。

协调者将在你完成自然评分和排序后，把前三名统一映射为 Major Revision、后两名统一映射为 Reject。评分阶段仍不得受名额影响。
"""
    (ROOM / "REVIEW_TASK.md").write_text(task, encoding="utf-8")


if __name__ == "__main__":
    main()
