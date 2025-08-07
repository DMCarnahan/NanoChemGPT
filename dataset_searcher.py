"""
dataset_searcher.py
====================

This module provides a file-agnostic mechanism for loading tabular data from a
variety of common formats (CSV, Excel, JSON array, JSON Lines) and performing
simple substring or regular-expression searches over the resulting table.

It is deliberately kept lightweight: no external dependencies beyond
``pandas`` are required. The returned objects are plain ``pandas.DataFrame``
instances, so downstream callers can manipulate the data further or convert it
to dictionaries as needed.

Example usage::

    from dataset_searcher import load_table, DatasetSearcher
    df = load_table("data/coremof.jsonl")
    searcher = DatasetSearcher(df)
    hits = searcher.query("UiO-66", topk=5)
    for _, row in hits.iterrows():
        print(row.to_dict())

"""

from __future__ import annotations

import json
import gzip
import lzma
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _read_json_array(fp: str | Path, comp: Optional[str] = None) -> pd.DataFrame:
    """
    Load a JSON file containing a top-level array of objects.

    Parameters
    ----------
    fp : str or Path
        Path to the JSON file.
    comp : str, optional
        Compression type if the file is compressed; one of ``"gzip"`` or
        ``"xz"``. If ``None``, the file is assumed to be uncompressed.

    Returns
    -------
    pandas.DataFrame
        A flattened DataFrame of the objects in the JSON array.
    """
    opener = gzip.open if comp == "gzip" else lzma.open if comp == "xz" else open
    with opener(fp, "rt") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("records", data)
    return pd.json_normalize(records)


def _read_jsonl(fp: str | Path, comp: Optional[str] = None) -> pd.DataFrame:
    """
    Load a JSON Lines (JSONL) file into a DataFrame.

    Parameters
    ----------
    fp : str or Path
        Path to the JSON Lines file.
    comp : str, optional
        Compression type; one of ``"gzip"`` or ``"xz"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each line of the file becomes one record.
    """
    opener = gzip.open if comp == "gzip" else lzma.open if comp == "xz" else open
    with opener(fp, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.json_normalize(rows)


def load_table(path: str | Path) -> pd.DataFrame:
    """
    Read a tabular dataset from the given file.

    Supported formats include CSV/CSV.GZ, Excel (.xlsx), JSON array,
    JSON Lines (.jsonl), and their gzip or xz-compressed variants.

    Parameters
    ----------
    path : str or Path
        Path to the file on disk.

    Returns
    -------
    pandas.DataFrame
        The loaded table.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    p = Path(path)
    suf = p.suffix.lower()

    match suf:
        case ".csv":
            return pd.read_csv(p)
        case ".csv.gz":
            return pd.read_csv(p)
        case ".xlsx":
            return pd.read_excel(p)
        case ".json":
            return _read_json_array(p)
        case ".jsonl":
            return _read_jsonl(p)
        case ".json.gz":
            return _read_json_array(p, "gzip")
        case ".jsonl.gz":
            return _read_jsonl(p, "gzip")
        case ".json.xz":
            return _read_json_array(p, "xz")
        case ".jsonl.xz":
            return _read_jsonl(p, "xz")
        case _:
            raise ValueError(f"Unsupported table type: {suf}")


class DatasetSearcher:
    """
    Simple keyword searcher over a Pandas DataFrame.

    Parameters
    ----------
    table : pandas.DataFrame
        The table to search.
    text_cols : iterable of str, optional
        Specific columns to search; if ``None``, all object-dtype columns are used.

    Notes
    -----
    Searches are case-insensitive by default and use substring matching.
    To perform regular-expression searches or case-sensitive searches, set the
    ``regex`` and ``case`` parameters on :meth:`query`.
    """

    def __init__(self, table: pd.DataFrame, text_cols: Optional[Iterable[str]] = None) -> None:
        self.df: pd.DataFrame = table
        if text_cols is not None:
            self.text_cols = list(text_cols)
        else:
            # infer columns of type object (strings)
            self.text_cols = [c for c in table.columns if table[c].dtype == "object"]

    def query(
        self,
        pattern: str,
        regex: bool = False,
        case: bool = False,
        topk: int = 10,
    ) -> pd.DataFrame:
        """
        Return rows where any of the designated text columns match ``pattern``.

        Parameters
        ----------
        pattern : str
            The search string or regular expression.
        regex : bool, default False
            If ``True``, treat ``pattern`` as a regular expression. If ``False``,
            perform a substring search.
        case : bool, default False
            If ``True``, searches are case-sensitive.
        topk : int, default 10
            Maximum number of rows to return.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing up to ``topk`` matching rows.
        """
        if not pattern:
            # return an empty DataFrame with the same columns when no pattern is provided
            return self.df.head(0)
        # compile the pattern only once
        pat = re.compile(pattern, 0 if case else re.IGNORECASE) if regex else pattern
        mask = pd.Series(False, index=self.df.index)
        for col in self.text_cols:
            # convert to string before searching
            series = self.df[col].astype(str)
            if regex:
                mask |= series.str.contains(pat)
            else:
                mask |= series.str.contains(pat, case=case, regex=False)
        return self.df[mask].head(topk).copy()