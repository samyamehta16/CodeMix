#!/usr/bin/env python3
"""
Fill the official BCSE497J Project-I template with TruthGuard Review content.

Placeholders still needing student input are marked [TO FILL: ...].
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from bcse497j_expanded_body import add_full_body

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "BCSE497J Project I Report - Template FINAlL Updated (2).docx"
OUT = ROOT / "docs" / "BCSE497J_Project_I_Report_TruthGuard.docx"
IMAGES = ROOT / "images"

# ---------------------------------------------------------------------------
# Known team details — change [TO FILL] items when you have them
# ---------------------------------------------------------------------------
PROJECT_TITLE = "CODE-MIXED NLP SYSTEMS FOR MULTILINGUAL COMMUNICATION"
PROJECT_TITLE_SHORT = "Code-Mixed NLP Systems for Multilingual Communication"

# Sorted by register number (as template requires)
STUDENTS = [
    ("23BCE0504", "PRANAV KALRA"),
    ("23BDS0204", "SAMYA MEHTA"),
    ("23BDS0209", "PALAK GOYAL"),
]

PROGRAMME = "Computer Science and Engineering with Specialization in Data Science"
GUIDE_NAME = "Dr. Sanjiban Sekhar Roy"
GUIDE_DESIGNATION = "Professor Grade 2"
HOD_NAME = "Dr. Sendhil Kumar K. S."
HOD_DESIGNATION = "Professor Grade 1"
DEPARTMENT_NAME = "School of Computer Science and Engineering (SCOPE)"
DEAN_NAME = "Dr. Jaisankar N"
REPORT_MONTH = "October 2026"
PLACE = "Vellore"
SIGN_DATE = "20 October 2026"


def set_run_font(run, size=12, bold=False, italic=False, name="Times New Roman", color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def clear_paragraph(p):
    p.text = ""


def set_paragraph_text(p, text, *, size=12, bold=False, italic=False, center=False, highlight=False):
    clear_paragraph(p)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    if highlight:
        rPr = run._element.get_or_add_rPr()
        hl = rPr.find(qn("w:highlight"))
        if hl is None:
            hl = rPr.makeelement(qn("w:highlight"), {qn("w:val"): "yellow"})
            rPr.append(hl)
        else:
            hl.set(qn("w:val"), "yellow")
    return p


def add_body(doc, text, *, size=12, space_after=8, first_indent=True):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.line_spacing = 1.15
    if first_indent:
        para.paragraph_format.first_line_indent = Inches(0.3)
    run = para.add_run(text)
    set_run_font(run, size=size)
    return para


def add_heading_line(doc, text, *, size=14, bold=True, center=False, space_before=14):
    para = doc.add_paragraph()
    if center:
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after = Pt(8)
    para.paragraph_format.line_spacing = 1.5
    run = para.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return para


def add_bullet(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.25)
    para.paragraph_format.space_after = Pt(3)
    para.paragraph_format.line_spacing = 1.15
    run = para.add_run(f"• {text}")
    set_run_font(run, size=12)
    return para


def add_caption(doc, text, *, above=False):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(4 if not above else 10)
    para.paragraph_format.space_after = Pt(10 if not above else 4)
    run = para.add_run(text)
    set_run_font(run, size=10, italic=True)
    return para


def add_table(doc, headers, rows, widths=None):
    # table caption should come ABOVE — caller adds caption first
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_run_font(run, size=10, bold=True)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            set_run_font(run, size=10)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return table


def add_picture(doc, path: Path, width=5.6):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    run.add_picture(str(path), width=Inches(width))
    return para


def replace_all_in_paragraph(p, mapping: dict[str, str]):
    # naive whole-paragraph replace if any key present
    text = p.text
    if not text:
        return False
    new = text
    for a, b in mapping.items():
        if a in new:
            new = new.replace(a, b)
    if new != text:
        # preserve basic bold of first run if any
        bold = p.runs[0].bold if p.runs else False
        size = p.runs[0].font.size.pt if p.runs and p.runs[0].font.size else 12
        set_paragraph_text(p, new, size=size, bold=bool(bold))
        return True
    return False


def delete_paragraph(p):
    el = p._element
    parent = el.getparent()
    if parent is not None:
        parent.remove(el)


def fill_front_matter(doc: Document):
    # Title
    set_paragraph_text(doc.paragraphs[2], PROJECT_TITLE, size=18, bold=True, center=True)

    # Remove yellow instruction under title if present
    if "Times New Roman 18" in doc.paragraphs[3].text:
        set_paragraph_text(doc.paragraphs[3], "", size=12)

    # Programme
    set_paragraph_text(doc.paragraphs[12], PROGRAMME, size=12, bold=True, center=True)

    # Instruction under names
    if "mention names" in doc.paragraphs[16].text.lower():
        set_paragraph_text(doc.paragraphs[16], "", size=12)

    # Student table (sorted by reg no.)
    st = doc.tables[0]
    for i, (reg, name) in enumerate(STUDENTS):
        st.rows[i].cells[0].text = ""
        r0 = st.rows[i].cells[0].paragraphs[0].add_run(reg)
        set_run_font(r0, size=12, bold=True)
        st.rows[i].cells[1].text = ""
        r1 = st.rows[i].cells[1].paragraphs[0].add_run(name)
        set_run_font(r1, size=12, bold=True)

    # Guide table
    gt = doc.tables[1]
    gt.rows[0].cells[0].text = ""
    g0 = gt.rows[0].cells[0].paragraphs[0].add_run(GUIDE_NAME)
    set_run_font(g0, size=12, bold=True)
    gt.rows[1].cells[0].text = ""
    g1 = gt.rows[1].cells[0].paragraphs[0].add_run(GUIDE_DESIGNATION)
    set_run_font(g1, size=12)
    # highlight designation if still TO FILL
    if GUIDE_DESIGNATION.startswith("[TO FILL"):
        rPr = g1._element.get_or_add_rPr()
        hl = rPr.makeelement(qn("w:highlight"), {qn("w:val"): "yellow"})
        rPr.append(hl)
    gt.rows[2].cells[0].text = ""
    g2 = gt.rows[2].cells[0].paragraphs[0].add_run(DEPARTMENT_NAME)
    set_run_font(g2, size=12)

    # Month
    set_paragraph_text(doc.paragraphs[23], REPORT_MONTH, size=12, bold=True, center=True)

    # Declaration
    decl = (
        f"I hereby declare that the project entitled {PROJECT_TITLE_SHORT} submitted by me, "
        f"for the award of the degree of Bachelor of Technology in {PROGRAMME} "
        f"to VIT is a record of bonafide work carried out by me under the supervision of {GUIDE_NAME}. "
        f"I further declare that the work reported in this project has not been submitted and will not "
        f"be submitted, either in part or in full, for the award of any other degree or diploma in this "
        f"institute or any other institute or university."
    )
    # Keep following sentence if template had two paras — overwrite para 27 fully
    set_paragraph_text(doc.paragraphs[27], decl, size=12)
    if "further declare" in doc.paragraphs[28].text.lower() or doc.paragraphs[28].text.strip() == "":
        # template may have second declare para already merged; clear duplicate if present
        if "further declare" in (doc.paragraphs[28].text or "").lower():
            set_paragraph_text(doc.paragraphs[28], "", size=12)

    set_paragraph_text(
        doc.paragraphs[36],
        f"Place\t: {PLACE}\t\tDate\t: {SIGN_DATE}",
        size=12,
    )

    # Remove declaration note
    if "Note: Include your project title" in doc.paragraphs[43].text:
        set_paragraph_text(doc.paragraphs[43], "", size=10)

    # Certificate
    student_list = ", ".join([f"{n} ({r})" for r, n in STUDENTS])
    cert = (
        f"This is to certify that the project entitled {PROJECT_TITLE_SHORT} submitted by "
        f"{student_list}, {DEPARTMENT_NAME}, VIT, for the award of the degree of Bachelor of "
        f"Technology in {PROGRAMME}, is a record of bonafide work carried out "
        f"by them under my supervision."
    )
    set_paragraph_text(doc.paragraphs[47], cert, size=12)
    set_paragraph_text(doc.paragraphs[55], f"Place\t: {PLACE}", size=12)
    set_paragraph_text(doc.paragraphs[56], f"Date\t: {SIGN_DATE}", size=12)

    if "Note: Include your project title" in doc.paragraphs[69].text:
        set_paragraph_text(doc.paragraphs[69], "", size=10)

    set_paragraph_text(
        doc.paragraphs[73],
        f"{HOD_NAME}\n{HOD_DESIGNATION}\n{DEPARTMENT_NAME}",
        size=12,
        bold=True,
    )
    set_paragraph_text(doc.paragraphs[74], PROGRAMME, size=12)

    # Acknowledgements
    ack_parts = {
        "Dr. Jaisankar N": DEAN_NAME,
        "<Head of Department’s Name>": HOD_NAME,
        "<Department Name>": DEPARTMENT_NAME,
        "<Guide’s Name>": GUIDE_NAME,
    }
    for idx in (79, 81, 83, 85, 87):
        replace_all_in_paragraph(doc.paragraphs[idx], ack_parts)
    # Rewrite HOD and guide ack paragraphs cleanly
    set_paragraph_text(
        doc.paragraphs[83],
        f"I express my profound appreciation to {HOD_NAME}, {HOD_DESIGNATION} and Head of the "
        f"{DEPARTMENT_NAME}, for insightful guidance and continuous support. Their expertise and "
        f"advice have been crucial in shaping the direction of this work.",
        size=12,
    )
    set_paragraph_text(
        doc.paragraphs[85],
        f"I am immensely thankful to my project guide, {GUIDE_NAME}, for dedicated mentorship and "
        f"invaluable feedback. Their patience, knowledge, and encouragement have been pivotal in "
        f"the successful completion of this project.",
        size=12,
    )
    set_paragraph_text(
        doc.paragraphs[91],
        "\n".join([n.title() for _, n in STUDENTS]),
        size=12,
    )

    # TOC table — fill contents/page estimates (roman/arabic approximate)
    toc = doc.tables[2]
    toc_rows = [
        ("", "Abstract", "i"),
        ("1.", "INTRODUCTION", "1"),
        ("", "1.1 Background", "1"),
        ("", "1.2 Motivations", "2"),
        ("", "1.3 Scope of the Project", "2"),
        ("2.", "PROJECT DESCRIPTION AND GOALS", "3"),
        ("", "2.1 Literature Review", "3"),
        ("", "2.2 Research Gap", "4"),
        ("", "2.3 Objectives", "4"),
        ("", "2.4 Problem Statement", "5"),
        ("", "2.5 Project Plan", "5"),
        ("3.", "TECHNICAL SPECIFICATION", "6"),
        ("", "3.1 Requirements", "6"),
        ("", "3.1.1 Functional", "6"),
        ("", "3.1.2 Non-Functional", "6"),
        ("", "3.2 Feasibility Study", "7"),
        ("", "3.2.1 Technical Feasibility", "7"),
        ("", "3.2.2 Economic Feasibility", "7"),
        ("", "3.2.3 Social Feasibility", "7"),
        ("", "3.3 System Specification", "8"),
        ("", "3.3.1 Hardware Specification", "8"),
        ("", "3.3.2 Software Specification", "8"),
        ("4.", "DESIGN APPROACH AND DETAILS", "9"),
        ("", "4.1 System Architecture", "9"),
        ("", "4.2 Design", "10"),
        ("", "4.2.1 Data Flow Diagram", "10"),
        ("", "4.2.2 Use Case Diagram", "10"),
        ("", "4.2.3 Class Diagram", "11"),
        ("", "4.2.4 Sequence Diagram", "11"),
        ("5.", "METHODOLOGY AND TESTING", "12"),
        ("", "5.1 Module Description", "12"),
        ("", "5.2 Testing", "14"),
        ("6.", "PROJECT DEMONSTRATION", "15"),
        ("7.", "RESULT AND DISCUSSION", "18"),
        ("8.", "CONCLUSION", "21"),
        ("9.", "REFERENCES", "22"),
        ("", "APPENDIX A – SAMPLE CODE", "23"),
    ]
    # Ensure enough rows
    while len(toc.rows) < 1 + len(toc_rows):
        toc._tbl.append(copy.deepcopy(toc.rows[-1]._tr))
    for i, (sl, content, page) in enumerate(toc_rows):
        row = toc.rows[i + 1]
        vals = [sl, content, page]
        for ci, val in enumerate(vals):
            row.cells[ci].text = ""
            run = row.cells[ci].paragraphs[0].add_run(val)
            bold = (sl.endswith(".") and sl[0].isdigit()) or content.isupper() or content == "Abstract"
            set_run_font(run, size=11, bold=bold)

    # List of figures table
    fig = doc.tables[3]
    figures = [
        ("4.1", "TruthGuard dual-pipeline architecture (Approach A + Approach B)", "9"),
        ("6.1", "Streamlit demo — FAKE verdict (RBI ₹5000 rumour)", "15"),
        ("6.2", "Streamlit demo — REAL verdict (Chandrayaan-3)", "16"),
        ("6.3", "Streamlit demo — MISLEADING verdict (school holiday claim)", "16"),
        ("6.4", "Streamlit demo — UNVERIFIED verdict (free iPhone claim)", "17"),
    ]
    while len(fig.rows) < 1 + len(figures):
        fig._tbl.append(copy.deepcopy(fig.rows[-1]._tr))
    for i, rowvals in enumerate(figures):
        row = fig.rows[i + 1]
        for ci, val in enumerate(rowvals):
            row.cells[ci].text = ""
            run = row.cells[ci].paragraphs[0].add_run(val)
            set_run_font(run, size=11)

    # List of tables paragraph area — update sample lines if present
    for p in doc.paragraphs:
        if p.text.strip().startswith("2.1") and "Table caption" in p.text:
            set_paragraph_text(
                p,
                "7.1\tHard-split classifier results (seed 42)\t18\n"
                "7.2\tRandom-split diagnostic results\t19\n"
                "3.1\tHardware / software specification\t8\n"
                "5.1\tApproach A models compared\t13",
                size=11,
            )

    # Abbreviations — replace sample mobile-network list
    abbr_start = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "2G\tSecond Generation":
            abbr_start = i
            break
    abbrs = [
        ("API", "Application Programming Interface"),
        ("BERT", "Bidirectional Encoder Representations from Transformers"),
        ("BM25", "Best Matching 25 (ranking function)"),
        ("FAKE / REAL", "Binary veracity labels used in Approach A"),
        ("LLM", "Large Language Model"),
        ("mBERT", "Multilingual BERT"),
        ("MuRIL", "Multilingual Representations for Indian Languages"),
        ("OOD", "Out-of-Distribution"),
        ("RAG", "Retrieval-Augmented Generation"),
        ("RBI", "Reserve Bank of India"),
        ("TF-IDF", "Term Frequency–Inverse Document Frequency"),
        ("Tanglish", "Tamil–English code-mixed text"),
        ("XLM-R", "XLM-RoBERTa"),
    ]
    if abbr_start is not None:
        # overwrite next ~7 sample abbr lines
        for j, (a, b) in enumerate(abbrs):
            idx = abbr_start + j
            if idx < len(doc.paragraphs) and (
                "\t" in doc.paragraphs[idx].text or doc.paragraphs[idx].text.strip() == ""
            ):
                set_paragraph_text(doc.paragraphs[idx], f"{a}\t{b}", size=12)
        # clear remaining sample mobile abbrs until note
        k = abbr_start + len(abbrs)
        while k < len(doc.paragraphs) and "Alphabetical order" not in doc.paragraphs[k].text:
            if doc.paragraphs[k].text.strip() and not doc.paragraphs[k].text.startswith("Symbols"):
                # stop if we hit Symbols
                if "Symbols" in doc.paragraphs[k].text:
                    break
                set_paragraph_text(doc.paragraphs[k], "", size=12)
            k += 1

    # Symbols — simplify for this project
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip().startswith("Symbols and Notations"):
            # next two sample symbol lines
            if i + 1 < len(doc.paragraphs):
                set_paragraph_text(
                    doc.paragraphs[i + 1],
                    "x\tInput text (headline or Tanglish claim)\n"
                    "c\tNormalized claim extracted from x\n"
                    "E\tRetrieved evidence set {e1..ek}\n"
                    "v\tFinal verdict label",
                    size=12,
                )
            if i + 2 < len(doc.paragraphs) and "NCFO" in doc.paragraphs[i + 2].text:
                set_paragraph_text(doc.paragraphs[i + 2], "", size=12)
            break


def remove_from_abstract_body(doc: Document):
    """Remove template stub from ABSTRACT heading through end (body rewritten next)."""
    abs_idx = None
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "ABSTRACT":
            abs_idx = i
            break
    if abs_idx is None:
        raise RuntimeError("ABSTRACT heading not found")

    to_delete = list(doc.paragraphs[abs_idx:])
    for p in to_delete:
        delete_paragraph(p)


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE, OUT)
    doc = Document(OUT)
    fill_front_matter(doc)
    remove_from_abstract_body(doc)
    add_full_body(doc)
    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Programme: {PROGRAMME}")
    print(f"Guide: {GUIDE_NAME}, {GUIDE_DESIGNATION}")
    print(f"HoD: {HOD_NAME}, {HOD_DESIGNATION}")
    print(f"Date: {SIGN_DATE}")


if __name__ == "__main__":
    main()
