#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update phenobox_data.csv so each row can be matched to DeepSDF / mesh sample ids (S<N>).

Inputs:
  - phenobox csv: /home/mainuser/mount/sungjay/kmelon_data/csvs/phenobox_data.csv
  - link map:     /home/mainuser/mount/sungjay/tcore/corepp/data/3DKmelonDemo/2_sfm/1_mesh/link_map_mesh.csv

Rules:
  - May/June (legacy) samples with id 1..70 map to S1..S70
    (user confirmed sample 1..70 identical to S1..S70 for May/June data)
  - Aug/Nov (and other) samples map via link_map_mesh.csv:
        sample_id == <dateDigits>_<idInt>  -> S_number / dest_name

Output:
  - Updates phenobox_data.csv in-place (after writing a timestamped backup),
    adding columns: sample_id, S_number, S_name

Safe:
  - Does not touch any DeepSDF experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import zipfile
import xml.etree.ElementTree as ET
import datetime as _dt
import math as _math
from pathlib import Path
from typing import Dict, List, Tuple


PHENO_DEFAULT = Path("/home/mainuser/mount/sungjay/kmelon_data/csvs/phenobox_data.csv")
LINK_DEFAULT = Path(
    "/home/mainuser/mount/sungjay/tcore/corepp/data/3DKmelonDemo/2_sfm/1_mesh/link_map_mesh.csv"
)


def _is_zip_xlsx(path: Path) -> bool:
    b = path.read_bytes()[:4]
    # XLSX files are ZIP archives starting with PK..
    return b.startswith(b"PK")


def _col_letters_to_index(col: str) -> int:
    """
    Convert Excel column letters (A, B, ..., Z, AA, AB, ...) to 0-based index.
    """
    n = 0
    for ch in col:
        if not ("A" <= ch <= "Z"):
            break
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def _xlsx_first_sheet_xml(z: zipfile.ZipFile) -> str:
    """
    Resolve the first worksheet path from workbook.xml + rels.
    Fallback to xl/worksheets/sheet1.xml if resolution fails.
    """
    try:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheets = wb.find("m:sheets", ns)
        if sheets is None:
            raise KeyError
        first_sheet = sheets.find("m:sheet", ns)
        if first_sheet is None:
            raise KeyError
        rid = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if not rid:
            raise KeyError
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        for rel in rels.findall("r:Relationship", rns):
            if rel.attrib.get("Id") == rid:
                target = rel.attrib.get("Target", "")
                # target is relative to xl/
                if target.startswith("/"):
                    target = target.lstrip("/")
                if not target.startswith("xl/"):
                    target = "xl/" + target
                return target
    except Exception:
        pass
    return "xl/worksheets/sheet1.xml"


def _read_xlsx_as_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Minimal XLSX reader for table-like data. Assumes first row is header.
    """
    with zipfile.ZipFile(path, "r") as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in sst.findall("m:si", ns):
                # concatenate all <t> nodes (handles rich text)
                texts = [t.text or "" for t in si.findall(".//m:t", ns)]
                shared.append("".join(texts))

        sheet_path = _xlsx_first_sheet_xml(z)
        sheet = ET.fromstring(z.read(sheet_path))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        # Build sparse row dicts: {row_idx: {col_idx: value}}
        table: Dict[int, Dict[int, str]] = {}
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            r_attr = row.attrib.get("r")
            if not r_attr:
                continue
            r_idx = int(r_attr) - 1  # 0-based row index
            row_map: Dict[int, str] = {}
            for c in row.findall("m:c", ns):
                ref = c.attrib.get("r", "")
                m = re.match(r"^([A-Z]+)", ref)
                if not m:
                    continue
                col_idx = _col_letters_to_index(m.group(1))
                v = c.find("m:v", ns)
                if v is None or v.text is None:
                    continue
                cell_type = c.attrib.get("t")
                if cell_type == "s":
                    # shared string
                    try:
                        row_map[col_idx] = shared[int(v.text)]
                    except Exception:
                        row_map[col_idx] = v.text
                else:
                    row_map[col_idx] = v.text
            if row_map:
                table[r_idx] = row_map

        if 0 not in table:
            raise SystemExit("Could not find header row in XLSX (row 1)")

        # Determine column count from header row
        header_row = table[0]
        max_col = max(header_row.keys())
        header = [header_row.get(i, "").strip() for i in range(max_col + 1)]
        if not any(header):
            raise SystemExit("Header row is empty in XLSX")

        rows: List[Dict[str, str]] = []
        for r_idx in sorted(k for k in table.keys() if k != 0):
            row_map = table[r_idx]
            vals = [row_map.get(i, "") for i in range(max_col + 1)]
            # skip completely empty rows
            if not any(str(v).strip() for v in vals):
                continue
            d = {header[i]: str(vals[i]) for i in range(len(header))}
            rows.append(d)

        # Remove empty header names (can happen if trailing empty columns)
        header_clean = [h for h in header if h.strip()]
        # Re-map rows to clean header
        cleaned_rows: List[Dict[str, str]] = []
        for r in rows:
            cleaned_rows.append({k: r.get(k, "") for k in header_clean})

        return header_clean, cleaned_rows


def _normalize_date_digits(s: str) -> str:
    """
    Normalize a date cell into digits suitable for sample_id matching.

    Supports:
      - Excel serial dates (e.g. 45873 -> 20250804)
      - formatted strings (e.g. 2025-08-04 -> 20250804)
      - compact digits (e.g. 20250804, 250526)
    """
    raw = str(s or "").strip()
    # Handle dotted/slashed formats like "2025.8.4" / "2025-8-4" / "2025/08/04"
    # before falling back to digit stripping.
    parts = re.split(r"[^\d]+", raw)
    parts = [p for p in parts if p]
    if len(parts) == 3 and len(parts[0]) == 4:
        try:
            y = int(parts[0])
            m = int(parts[1])
            d = int(parts[2])
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                return f"{y:04d}{m:02d}{d:02d}"
        except Exception:
            pass
    # Excel serial date detection: pure number with <= 5 digits (or numeric string)
    # Excel base for Windows: 1899-12-30
    if raw and re.fullmatch(r"\d+(?:\.0+)?", raw):
        try:
            v = float(raw)
            if _math.isfinite(v):
                iv = int(round(v))
                # plausible range for modern dates
                if 30000 <= iv <= 70000:
                    base = _dt.date(1899, 12, 30)
                    d = base + _dt.timedelta(days=iv)
                    return d.strftime("%Y%m%d")
        except Exception:
            pass

    # Otherwise: keep only digits
    d = re.sub(r"\D+", "", raw)
    return d


def _intish(x: str) -> int | None:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _read_pheno_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    """
    Read phenobox table from either a real CSV or a mislabeled XLSX (zip).
    Returns (fieldnames, rows, source_kind).
    """
    if _is_zip_xlsx(path):
        header, rows = _read_xlsx_as_rows(path)
        return header, rows, "xlsx(zip)"

    # CSV path: try encodings; handle odd bytes gracefully
    candidates = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"]
    last_err: Exception | None = None
    for enc in candidates:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise SystemExit(f"No header found in {path}")
                rows = [row for row in reader]
                return list(reader.fieldnames), rows, f"csv({enc})"
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"Failed to read {path} as CSV. Last error: {last_err}")


def _write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_link_map(link_csv: Path) -> Dict[str, Tuple[str, str]]:
    """
    Returns mapping: sample_id -> (S_number_str, S_name)
    """
    mp: Dict[str, Tuple[str, str]] = {}
    with link_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = (row.get("sample_id") or "").strip()
            s_num = (row.get("S_number") or "").strip()
            s_name = (row.get("dest_name") or "").strip()
            if sid and s_name:
                mp[sid] = (s_num, s_name)
    return mp


def _find_col(fieldnames: List[str], candidates: List[str]) -> str | None:
    lower_to_real = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    # fuzzy contains match
    for real in fieldnames:
        rl = real.lower()
        for cand in candidates:
            if cand.lower() in rl:
                return real
    return None


def _find_col_exact(fieldnames: List[str], candidates: List[str]) -> str | None:
    """
    Exact (case-insensitive) match only. Useful to avoid fuzzy collisions like:
      'id' matching 'sample_id'.
    """
    lower_to_real = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        hit = lower_to_real.get(cand.lower())
        if hit is not None:
            return hit
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Add S-number mapping columns to phenobox_data.csv")
    ap.add_argument("--phenobox_csv", type=str, default=str(PHENO_DEFAULT))
    ap.add_argument("--link_map_csv", type=str, default=str(LINK_DEFAULT))
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Update phenobox_csv in-place (default). If false, writes <phenobox_csv>.updated.csv",
    )
    ap.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not write a backup before overwriting (NOT recommended).",
    )
    args = ap.parse_args()

    phenobox = Path(args.phenobox_csv)
    link_csv = Path(args.link_map_csv)
    if not phenobox.exists():
        raise SystemExit(f"phenobox csv not found: {phenobox}")
    if not link_csv.exists():
        raise SystemExit(f"link map csv not found: {link_csv}")

    fieldnames, rows, src_kind = _read_pheno_rows(phenobox)

    # locate required columns (date + id)
    date_col = _find_col(fieldnames, ["date", "Date", "측정일", "날짜"])
    # Prefer exact matches to avoid picking 'sample_id' when we really want 'sample'
    id_col = _find_col_exact(fieldnames, ["sample", "id", "ID"])
    if id_col is None:
        id_col = _find_col(fieldnames, ["sample", "id", "번호", "개체", "index"])
    if date_col is None or id_col is None:
        raise SystemExit(
            f"Could not find required columns in phenobox_data.csv. "
            f"Found date_col={date_col}, id_col={id_col}. Columns={fieldnames}"
        )

    link_map = _build_link_map(link_csv)

    # ensure columns exist (append at end)
    new_cols = ["sample_id", "S_number", "S_name"]
    out_fieldnames = fieldnames[:]
    for c in new_cols:
        if c not in out_fieldnames:
            out_fieldnames.append(c)

    updated = 0
    mapped_link = 0
    mapped_legacy = 0
    unmapped = 0

    for r in rows:
        date_digits = _normalize_date_digits(r.get(date_col, ""))
        sid_int = _intish(r.get(id_col, ""))

        sample_id = ""
        s_num = ""
        s_name = ""

        if date_digits and sid_int is not None:
            sample_id = f"{date_digits}_{sid_int}"

            # legacy May/June -> S1..S70
            is_may_june = False
            if len(date_digits) == 8:
                try:
                    mm = int(date_digits[4:6])
                    is_may_june = mm in (5, 6)
                except Exception:
                    is_may_june = False
            else:
                # handle yyMMdd (e.g. 250526) if it appears
                is_may_june = date_digits.startswith("2505") or date_digits.startswith("2506")

            if is_may_june and 1 <= sid_int <= 70:
                s_num = str(sid_int)
                s_name = f"S{sid_int}.ply"
                mapped_legacy += 1
            else:
                hit = link_map.get(sample_id)
                if hit:
                    s_num, s_name = hit
                    mapped_link += 1
                else:
                    unmapped += 1
        else:
            unmapped += 1

        r["sample_id"] = sample_id
        r["S_number"] = s_num
        r["S_name"] = s_name
        updated += 1

    # output
    if args.inplace or True:
        out_path = phenobox
    else:
        out_path = phenobox.with_suffix(phenobox.suffix + ".updated.csv")

    if out_path == phenobox and not args.no_backup:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = phenobox.with_suffix(phenobox.suffix + f".bak_{ts}")
        backup.write_bytes(phenobox.read_bytes())
        print(f"Wrote backup: {backup}")

    _write_csv_rows(out_path, out_fieldnames, rows)

    print(f"Read phenobox: {phenobox} ({src_kind}) rows={len(rows)}")
    print(f"Read link map: {link_csv} entries={len(link_map)}")
    print(f"Wrote updated: {out_path}")
    print(f"Mapped via link_map: {mapped_link}")
    print(f"Mapped via legacy May/June rule: {mapped_legacy}")
    print(f"Unmapped rows: {unmapped}")


if __name__ == "__main__":
    main()


