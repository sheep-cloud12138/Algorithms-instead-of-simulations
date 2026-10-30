from __future__ import annotations

import html
import re
import subprocess
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tmp" / "blind_review_source"
OUT = ROOT / "blind_review_packet"
PACKET_NAME = "blind_review_packet"


def pdf_text(path: Path, layout: bool = True) -> str:
    command = ["pdftotext"]
    if layout:
        command.append("-layout")
    command.extend([str(path), "-"])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def cut_pdf_body(text: str, start: str, stops: tuple[str, ...]) -> str:
    lines = text.splitlines()
    start_index = next(
        (i for i, line in enumerate(lines) if re.search(start, line, re.I)),
        None,
    )
    if start_index is None:
        raise ValueError(f"Could not find body start: {start}")
    stop_index = next(
        (
            i
            for i, line in enumerate(lines[start_index + 1 :], start_index + 1)
            if any(re.search(pattern, line, re.I) for pattern in stops)
        ),
        len(lines),
    )
    body = "\n".join(lines[start_index:stop_index])
    return clean_common_artifacts(body)


def clean_common_artifacts(text: str) -> str:
    text = text.replace("\f", "\n")
    text = re.sub(r"arXiv:[^\s]+(?:\s+\[[^\]]+\])?(?:\s+\d{1,2}\s+\w+\s+\d{4})?", "", text)
    text = re.sub(r"(?im)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"(?im)^\s*(?:page\s+)?\d+\s*(?:of|/)\s*\d+\s*$", "", text)
    text = re.sub(r"(?im)^\s*preprint submitted to[^\n]+$", "", text)
    text = re.sub(r"(?im)^\s*doi:\s*\S+\s*$", "", text)
    text = re.sub(r"(?im)^\s*corresponding author:[^\n]+$", "", text)
    text = re.sub(r"(?im)^\s*contact\s+[^\n]+$", "", text)
    text = re.sub(r"(?im)^\s*extended author information[^\n]*$", "", text)
    text = re.sub(r"(?im)^\s*©\s*\d{4}[^\n]*$", "", text)
    lines = text.splitlines()
    normalized = [re.sub(r"\s+", " ", line).strip() for line in lines]
    counts = Counter(line for line in normalized if line)
    lines = [
        line
        for line, norm in zip(lines, normalized)
        if not (norm and len(norm) < 200 and counts[norm] >= 5)
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def clean_html_body(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    sections = soup.find_all("section", class_=re.compile(r"ltx_section"))
    kept = []
    started = False
    for section in sections:
        heading = section.find("h2", class_=re.compile(r"ltx_title_section"))
        heading_text = heading.get_text(" ", strip=True) if heading else ""
        if not started and re.match(r"^\s*(?:\d+|[IVX]+)\s+Introduction\b", heading_text, re.I):
            started = True
        if not started:
            continue
        kept.append(section)
        if re.match(r"^\s*(?:\d+|[IVX]+)\s+(?:Conclusions?|Summary(?:\s+and\s+[A-Za-z ]+)?)\b", heading_text, re.I):
            break
    if not kept:
        raise ValueError(f"No body sections found in {path}")
    body = "\n\n".join(section.get_text("\n", strip=True) for section in kept)
    return clean_common_artifacts(html.unescape(body))


def redact_identifiers(text: str) -> str:
    text = re.sub(r"\[(?:\d+(?:\s*[-,–]\s*\d+)*)\]", "[citation]", text)
    text = re.sub(r"\[[^\]\n]{0,180}(?:19|20)\d{2}[^\]\n]*\]", "[citation]", text)
    text = re.sub(
        r"\((?:[A-Z][\w'’-]+(?:\s+(?:and|&)\s+[A-Z][\w'’-]+)?"
        r"(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}[a-z]?)\)",
        "[citation]",
        text,
    )
    text = re.sub(r"\[citation\]{2,}", "[citation]", text)
    text = re.sub(r"DOI\s*:\s*\S+", "[source identifier]", text, flags=re.I)
    text = re.sub(
        r"\b(?:arXiv|DOI)\s*:\s*[A-Za-z0-9./_-]+",
        "[source identifier]",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bHAL\s*:\s*[A-Za-z0-9./_-]+", "[source identifier]", text)
    text = re.sub(r"\b(?:https?|ftp)://\S+", "[URL]", text, flags=re.I)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    for name in (
        "Nurshat Menglik",
        "Menglik",
        "Alex Shao",
        "Shao",
        "David Hyde",
        "Hyde",
        "Mangnike",
        "Saeed Akbari",
        "Akbari",
        "Suraj Pawar",
        "Pawar",
        "Omer San",
        r"O\.\s*San",
        "San",
        "Liyuan Hu",
        "Shihe Xin",
        "Yvonne Chávez Chenaa",
        "Jalel Chergui",
        "Patrick Le Quéré",
        "Ao Xu",
        "Zheng Zhao",
        "Ben-Rui Xu",
        "Li-Sheng Jiang",
        "Fynn Fromme",
        "Fromme",
        "Hans Harder",
        "Christine Allen-Blanchette",
        "Allen-Blanchette",
        "Sebastian Peitz",
        "Peitz",
        "Shuang Wang",
        "Wanying Kang",
        "Sangam Khanal",
        "Khanal",
        "Shilaj Baral",
        "Baral",
        "Joongoo Jeon",
        "Ali Kashefi",
        "Kashefi",
        "Tapan Mukerji",
        "Mukerji",
        "Mohammed N. Alshehri",
        "Alshehri",
        "A. F. Aljohani",
        "Aljohani",
        "N. Ameer Ahammad",
        "Ioannis Kyritsopoulos",
        "Kyritsopoulos",
        "Saleh Rezaeiravesh",
        "Rezaeiravesh",
        "Vladimir Duffal",
        "Duffal",
        "Sofiane Benhamadouche",
        "Benhamadouche",
        "Alistair Revell",
        "Sree Pradip Kumer Sarker",
        "S. P. K. Sarker",
        "Sarker",
        "Md. Mahmud Alam",
        "M. M. Alam",
        "Andaç Batur Çolak",
        "A. Çolak",
        "Çolak",
        "David Barreiro-Villaverde",
        "D. Barreiro-Villaverde",
        "Barreiro-Villaverde",
        "David Barreiro",
        "Antonio Cantiani",
        "M. A. Mendez",
        "M.A. Mendez",
        "De Ryck",
        "Mishra",
        "Nguyen",
        "Vesper",
        "Rokicki",
        "Yong",
        "He and Luo",
        "Sun and Tao",
        "Luo",
        "Chen",
        "Zhao",
        "Xu",
        "Tao",
        "Yaji",
        "Barreiro",
    ):
        text = re.sub(re.escape(name), "[redacted name]", text, flags=re.I)
    text = re.sub(r"\bthe authors\b", "the manuscript", text, flags=re.I)
    return text


def write_paper(letter: str, title: str, body: str) -> None:
    body = redact_identifiers(body)
    header = (
        f"# 匿名稿件 {letter}\n\n"
        "> 评审材料已去除题名、作者、单位、摘要、关键词、致谢、基金、"
        "数据与代码链接、参考文献和来源标识。正文自引言开始，"
        "至结论结束。\n\n"
    )
    (OUT / f"paper_{letter}_body.md").write_text(header + body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("paper_*_body.md"):
        old.unlink()

    papers = {
        "A": (
            "local neural surrogate manuscript",
            cut_pdf_body(
                pdf_text(ROOT / "论文" / "manuscript_submission.pdf"),
                r"^\s*1\.\s*Introduction\b",
                (r"^\s*(?:Supplementary material|CRediT|Declaration|Data availability|References|Acknowledg)",),
            ),
        ),
        "B": (
            "buoyant convection with ANN analysis",
            cut_pdf_body(
                pdf_text(SOURCE / "fair_c_water.pdf", layout=False),
                r"^\s*Natural convection is a fundamental process\b",
                (r"^\s*(?:Funding|Data Availability|References)\b",),
            ),
        ),
        "C": (
            "data-driven transition-model calibration for natural convection",
            cut_pdf_body(
                pdf_text(SOURCE / "fair_d_calibration.pdf", layout=False),
                r"^\s*(?:1\.?\s+)?Introduction\b",
                (r"\bAcknowledgements?\b", r"^\s*(?:Open Access|Declarations|References)\b"),
            ),
        ),
        "D": (
            "machine-learning MHD natural convection in cavities",
            cut_pdf_body(
                pdf_text(SOURCE / "fair_d_scirp.pdf", layout=False),
                r"^\s*(?:1\.?\s+)?Introduction\b",
                (r"^\s*Acknowledgements\b", r"^\s*(?:Conflicts of Interest|References)\b"),
            ),
        ),
        "E": (
            "machine-learning entropy and heat transfer in a porous cavity",
            cut_pdf_body(
                pdf_text(SOURCE / "fair_e_rimni.pdf", layout=False),
                r"^\s*(?:1\.?\s+)?Introduction\b",
                (r"\bAcknowledgement\b", r"^\s*(?:Funding Statement|Availability of Data|References)\b"),
            ),
        ),
    }
    for letter, (_, body) in papers.items():
        write_paper(letter, "", body)

    instructions = """# 匿名独立评审

你是一位热流体力学与物理机器学习领域的资深审稿人。请只依据本目录中的五份匿名正文完成一次独立评审。

## 硬性限制

- 禁止联网，禁止调用网页、搜索、数据库或任何外部来源。
- 禁止读取本目录之外的任何文件，禁止推断或检索稿件的题名、作者、单位、期刊、年份、引用来源。
- 只使用每篇稿件给出的正文。不要使用目录名、文件名或写作风格猜测来源。
- 本次只做一轮评审，不要求作者回复。
- 先独立评分和排序，不得用任何配额影响分数。

## 评分

每篇给出总分 1-10（可保留 1 位小数）和以下 5 项各 1-5 分：

1. 科学重要性
2. 方法与证据
3. 创新增量
4. 可复现与稳健性
5. 表述与论证完整性

总分不是五项平均；请体现致命缺陷。

## 输出

先独立完成五维评分和排序，并只按分数确定名次。评分完成后，再把排名前三篇映射为 `Major Revision`、后两篇映射为 `Reject`；不得在评分阶段考虑这一配额。

排序表列出稿件编号、总分、各项分数和最终建议。随后逐篇给出：

- 一句话判断；
- 2-3 个主要优点（必须有正文证据）；
- 3-5 个决定建议的关键问题（必须具体、可回应）；
- 若为 Major Revision，列出必须完成的修改；
- 若为 Reject，说明为何在本次投稿中无法通过修改解决，或为何工作量超出正常大修范围。

最后给一段横向比较，说明三篇大修和两篇拒稿的分界依据。不要写客套话。
"""
    (OUT / "README_reviewer.md").write_text(instructions, encoding="utf-8")

    manifest = """# 匿名评审材料来源映射（私有，不提交给评审者）

| 匿名编号 | 来源 |
|---|---|
| A | 论文/manuscript_submission.pdf |
| B | Applied Water Science 2025, DOI 10.1007/s13201-025-02591-2 |
| C | Flow, Turbulence and Combustion 2026, DOI 10.1007/s10494-026-00770-y |
| D | Advances in Nanoparticles 2025, DOI 10.4236/anp.2025.144011 |
| E | Rev. Int. Mét. Numér. Cálc. Diseño Ing. 2026, DOI 10.23967/j.rimni.2025.10.71901 |

给评审者的目录中只含 `README_reviewer.md` 与五份 `paper_*_body.md`；本映射文件位于目录外。
"""
    (ROOT / "blind_review_source_manifest.md").write_text(manifest, encoding="utf-8")


if __name__ == "__main__":
    main()
