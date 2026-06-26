"""xlsx_kit — a tiny stdlib reader for the `.xlsx` supplementary tables the karyon probes ingest.

An `.xlsx` is a zip of XML, so a supplementary table parses with `zipfile` + `xml.etree` alone —
no `openpyxl`/`pandas`, keeping the probes' dependency-free posture (the same stance as `linmodel`
hand-rolling Cholesky and `stats_kit` hand-rolling Spearman). This is the SHARED extraction of the
reader first written inline in `promoter_data.py`; new loaders (`crispr_qc_data`) import it instead
of re-deriving the XML plumbing. `promoter_data.py` keeps its own inline copy for now (working code,
left untouched) and can adopt this later.

    from .xlsx_kit import workbook, sheet_names, rows
    z = workbook(raw_bytes)                       # raw = the downloaded .xlsx bytes
    for row in rows(z, "CRISPRi"):               # {col_letter: cell_text}, streamed, strings resolved
        seq = row.get("F")

Cells are returned as their raw text; numeric parsing is the caller's job (an empty cell is simply
absent from the row dict). Shared-string cells (`t="s"`) are resolved against the workbook table.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from xml.etree import ElementTree as ET

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def workbook(raw: bytes) -> zipfile.ZipFile:
    """The `.xlsx` bytes as an in-memory zip (it is a zip of XML). Raises `BadZipFile` if not."""
    return zipfile.ZipFile(io.BytesIO(raw))


def sheet_names(z: zipfile.ZipFile) -> list[str]:
    """The workbook's sheet display names, in workbook order."""
    wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
    return re.findall(r'<sheet[^>]*name="([^"]*)"', wb)


def shared_strings(z: zipfile.ZipFile) -> list[str]:
    """The shared-string table; a cell with `t='s'` carries an index into this list."""
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in root.iter(_NS + "si")]


def _sheet_path(z: zipfile.ZipFile, name: str) -> str:
    """The `xl/worksheets/sheetN.xml` member backing the sheet whose display name is `name`."""
    wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8", "replace")
    rid = dict(re.findall(r'<sheet[^>]*name="([^"]*)"[^>]*r:id="([^"]*)"', wb)).get(name)
    target = dict(re.findall(r'<Relationship[^>]*Id="([^"]*)"[^>]*Target="([^"]*)"', rels)).get(rid)
    if not target:
        raise KeyError(f"sheet {name!r} not found in workbook (format drift?)")
    return "xl/" + target.lstrip("/")


def _col(ref: str) -> str:
    """The column letters of a cell reference ('B12' -> 'B')."""
    return re.match(r"[A-Z]+", ref).group()


def rows(z: zipfile.ZipFile, sheet_name: str) -> Iterator[dict[str, str]]:
    """Yield each worksheet row of `sheet_name` as {col_letter: text}, streaming (files are big)."""
    ss = shared_strings(z)
    path = _sheet_path(z, sheet_name)
    with z.open(path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag != _NS + "row":
                continue
            out: dict[str, str] = {}
            for c in el.findall(_NS + "c"):
                v = c.find(_NS + "v")
                if v is None or v.text is None:
                    continue
                out[_col(c.get("r"))] = ss[int(v.text)] if c.get("t") == "s" else v.text
            el.clear()
            yield out
