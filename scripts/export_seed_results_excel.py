from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_SUMMARY = Path("results/seed_runs/seed_summary.csv")
DEFAULT_LONG = Path("results/seed_runs/seed_results_long.csv")
DEFAULT_OUTPUT = Path("results/seed_runs/seed_results.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export seed result CSVs to a lightweight XLSX workbook.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--long", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary_fields, summary_rows = _read_csv(args.summary)
    long_fields, long_rows = _read_csv(args.long)
    readme_fields = ["key", "value"]
    readme_rows = [
        {"key": "generated_utc", "value": datetime.now(timezone.utc).replace(microsecond=0).isoformat()},
        {"key": "summary_csv", "value": str(args.summary)},
        {"key": "long_csv", "value": str(args.long)},
        {"key": "note", "value": "Use rows with complete=true in summary for final reported seed aggregates."},
    ]
    write_seed_workbook(
        args.output,
        [
            ("summary", summary_fields, summary_rows),
            ("long", long_fields, long_rows),
            ("README", readme_fields, readme_rows),
        ],
    )
    print(f"wrote {args.output}")


def write_seed_workbook(path: Path, sheets: list[tuple[str, list[str], list[dict[str, str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = [_clean_sheet_name(name, index) for index, (name, _, _) in enumerate(sheets, start=1)]
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _content_types(len(sheets)))
        zf.writestr("_rels/.rels", _root_rels())
        zf.writestr("xl/workbook.xml", _workbook_xml(sheet_names))
        zf.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(sheets)))
        zf.writestr("xl/styles.xml", _styles_xml())
        for index, (_, fields, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{index}.xml", _sheet_xml(fields, rows))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _content_types(n_sheets: int) -> str:
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n_sheets + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{sheet_overrides}'
        '</Types>'
    )


def _root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheets}</sheets>'
        '</workbook>'
    )


def _workbook_rels(n_sheets: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n_sheets + 1)
    )
    rels += (
        f'<Relationship Id="rId{n_sheets + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{rels}'
        '</Relationships>'
    )


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        '</styleSheet>'
    )


def _sheet_xml(fields: list[str], rows: list[dict[str, str]]) -> str:
    all_rows: list[list[str]] = [fields]
    all_rows.extend([[row.get(field, "") for field in fields] for row in rows])
    max_col = max(1, len(fields))
    max_row = max(1, len(all_rows))
    rows_xml = []
    for row_index, values in enumerate(all_rows, start=1):
        cells = []
        for col_index, value in enumerate(values, start=1):
            style = ' s="1"' if row_index == 1 else ""
            cells.append(_cell_xml(row_index, col_index, value, style=style))
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    cols_xml = "".join(
        f'<col min="{i}" max="{i}" width="{_column_width(fields[i - 1] if i <= len(fields) else "")}" customWidth="1"/>'
        for i in range(1, max_col + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{_col_name(max_col)}{max_row}"/>'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols_xml}</cols>'
        f'<sheetData>{"".join(rows_xml)}</sheetData>'
        '</worksheet>'
    )


def _cell_xml(row_index: int, col_index: int, value: str, *, style: str = "") -> str:
    ref = f"{_col_name(col_index)}{row_index}"
    text = "" if value is None else str(value)
    if text == "":
        return f'<c r="{ref}"{style}/>'
    if _is_number(text):
        return f'<c r="{ref}"{style}><v>{escape(text)}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style}><is><t>{escape(text)}</t></is></c>'


def _is_number(value: str) -> bool:
    if value.strip() != value or value == "":
        return False
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", value) is None:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _column_width(header: str) -> int:
    return min(max(len(header) + 2, 10), 36)


def _clean_sheet_name(name: str, index: int) -> str:
    clean = re.sub(r"[\\/*?:\[\]]", "_", name).strip() or f"Sheet{index}"
    return clean[:31]


if __name__ == "__main__":
    main()
