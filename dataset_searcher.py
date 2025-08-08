from __future__ import annotations

import csv
import json
import gzip
import lzma
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, List

import pandas as pd


# ---------------- I/O helpers ----------------
def _normalize_records(obj):
    """Return a list of dict-like records from common JSON structures."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    # Some files use { "records": [...] } or { "data": [...] }
    for key in ("records", "data", "items", "rows"):
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    # Fallback: wrap object in list
    return [obj]


def _open_text(fp: str | Path, comp: Optional[str]) -> object:
    if comp == "gzip":
        return gzip.open(fp, "rt", encoding="utf-8", errors="replace")
    if comp == "xz":
        return lzma.open(fp, "rt", encoding="utf-8", errors="replace")
    return open(fp, "rt", encoding="utf-8", errors="replace")


def _read_json_array(fp: str | Path, comp: Optional[str] = None) -> pd.DataFrame:
    with _open_text(fp, comp) as f:
        data = json.load(f)
    return pd.json_normalize(_normalize_records(data), sep=".")


def _read_jsonl(fp: str | Path, comp: Optional[str] = None) -> pd.DataFrame:
    rows: List[dict] = []
    with _open_text(fp, comp) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                # best-effort: ignore bad lines
                continue
    return pd.json_normalize(rows, sep=".")


def _infer_suffixes(p: Path):
    """Return (base_suffix, compression) from a path with possibly multiple suffixes."""
    suff = [s.lower() for s in p.suffixes]  # e.g. ['.json', '.gz']
    comp = None
    if suff and suff[-1] in (".gz", ".gzip"):
        comp = "gzip"
        suff = suff[:-1]
    elif suff and suff[-1] in (".xz", ".lzma"):
        comp = "xz"
        suff = suff[:-1]
    ext = "".join(suff[-2:]) if suff and suff[-1] in (".json", ".jsonl", ".csv", ".tsv") and len(suff)>1 else (suff[-1] if suff else "")
    # ext is like '.jsonl', '.csv', '.tsv', '.json', '.xlsx', '.parquet'
    if not ext and p.suffix:
        ext = p.suffix.lower()
    return ext, comp


def load_table(path: str | Path, *, encoding: str = "utf-8", errors: str = "replace", **csv_kwargs) -> pd.DataFrame:
    """
    Read a tabular dataset from *path*.

    Supported formats:
      - CSV/CSV.GZ/CSV.XZ
      - TSV/TSV.GZ/TSV.XZ
      - Excel (.xlsx)
      - Parquet (.parquet)  (requires pyarrow or fastparquet)
      - JSON array (.json)  + compressed variants
      - JSON Lines (.jsonl) + compressed variants

    Parameters
    ----------
    path : str | Path
        File path
    encoding : str
        Text encoding for CSV/TSV/JSON (default 'utf-8')
    errors : str
        Error handling for decoding (default 'replace')
    **csv_kwargs :
        Extra args passed to pandas.read_csv / read_table (e.g., dtype, engine)

    Returns
    -------
    pandas.DataFrame
    """
    p = Path(path)
    ext, comp = _infer_suffixes(p)

    # Normalize for mixed case (e.g., .CSV, .Jsonl.GZ)
    ext = (ext or "").lower()

    # Choose reader
    if ext in (".csv",):
        return pd.read_csv(p, encoding=encoding, errors=errors, **csv_kwargs)
    if ext in (".tsv",):
        return pd.read_table(p, encoding=encoding, errors=errors, **csv_kwargs)
    if ext in (".xlsx",):
        return pd.read_excel(p)
    if ext in (".parquet",):
        try:
            return pd.read_parquet(p) 
        except Exception as e:
            raise RuntimeError("Reading parquet requires pyarrow or fastparquet. Install one and retry.") from e
    if ext in (".json",):
        return _read_json_array(p, comp)
    if ext in (".jsonl",):
        return _read_jsonl(p, comp)

    if p.suffix.lower() in (".csv", ".tsv"):
        sep = "," if p.suffix.lower() == ".csv" else "\\t"
        return pd.read_csv(p, sep=sep, encoding=encoding, errors=errors, **csv_kwargs)

    raise ValueError(f"Unsupported or unknown table type: {{p.name}} (ext='{{ext}}', comp='{{comp}}')")


# ---------------- Searcher ----------------
class DatasetSearcher:
    """Simple keyword/regex searcher over a Pandas DataFrame.

    Parameters
    ----------
    table : pandas.DataFrame
        The table to search.
    text_cols : iterable of str, optional
        Specific columns to search; if ``None``, all object-dtype columns are used.
    """
    def __init__(self, table: pd.DataFrame, text_cols: Optional[Iterable[str]] = None) -> None:
        self.df: pd.DataFrame = table
        if text_cols is not None:
            self.text_cols = [c for c in text_cols if c in table.columns]
        else:
            # infer likely text columns (object dtype or pandas string dtype)
            self.text_cols = [c for c in table.columns if str(table[c].dtype) in ("object", "string")]

    def _compile(self, pattern: str, case: bool, regex: bool):
        if not regex:
            return pattern  # plain substring
        flags = 0 if case else re.IGNORECASE
        try:
            return re.compile(pattern, flags)
        except re.error:
            # Fallback to a literal search if regex is invalid
            return re.compile(re.escape(pattern), flags)

    def query(
        self,
        pattern: str,
        regex: bool = False,
        case: bool = False,
        topk: int = 10,
        all_matches: bool = False,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return rows where any of the designated text columns match *pattern*.

        Parameters
        ----------
        pattern : str
            The search string or regular expression.
        regex : bool, default False
            If ``True``, treat *pattern* as a regular expression.
        case : bool, default False
            If ``True``, searches are case-sensitive.
        topk : int, default 10
            Maximum number of rows to return (ignored if *all_matches* is True).
        all_matches : bool, default False
            If True, return all matching rows (no truncation).
        columns : sequence of str, optional
            Subset of columns to search for this query (defaults to self.text_cols).

        Returns
        -------
        pandas.DataFrame
        """
        if not pattern:
            return self.df.head(0).copy()

        cols = [c for c in (columns or self.text_cols) if c in self.df.columns]
        if not cols:
            return self.df.head(0).copy()

        pat = self._compile(pattern, case=case, regex=regex)

        mask = pd.Series(False, index=self.df.index)
        for col in cols:
            s = self.df[col].astype("string").fillna("")
            if regex and hasattr(pat, "search"):
                mask |= s.str.contains(pat)
            else:
                mask |= s.str.contains(pat if regex else str(pat), case=case, regex=bool(regex))

        result = self.df[mask]
        if all_matches:
            return result.copy()
        return result.head(max(0, int(topk))).copy()


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Load a dataset and run a substring/regex search.")
    ap.add_argument("path", help="Path to dataset (csv/tsv/xlsx/parquet/json/jsonl, with optional .gz/.xz)")
    ap.add_argument("pattern", help="Search string or regex")
    ap.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    ap.add_argument("--case", action="store_true", help="Case-sensitive search")
    ap.add_argument("--topk", type=int, default=10, help="Max rows (ignored with --all)")
    ap.add_argument("--all", dest="all_matches", action="store_true", help="Return all matches")
    ap.add_argument("--columns", nargs="*", default=None, help="Limit search to these columns")
    ap.add_argument("--to", choices=["json","csv"], default="json", help="Output format")
    ap.add_argument("--encoding", default="utf-8", help="Encoding for CSV/TSV/JSON reads")
    ap.add_argument("--errors", default="replace", help="Decoding error policy (replace|ignore|strict)")
    args, rest = ap.parse_known_args()

    try:
        df = load_table(args.path, encoding=args.encoding, errors=args.errors)
        searcher = DatasetSearcher(df)
        out = searcher.query(
            args.pattern, regex=args.regex, case=args.case,
            topk=args.topk, all_matches=args.all_matches, columns=args.columns
        )
        if args.to == "csv":
            # write to stdout as CSV
            out.to_csv(sys.stdout, index=False)
        else:
            # json to stdout
            print(out.to_json(orient="records", force_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}), file=sys.stderr)
        sys.exit(1)
