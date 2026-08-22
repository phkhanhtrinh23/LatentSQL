"""SQLite schema introspection and prompt rendering.

Rendering follows DAIL-SQL's finding (section 3.1) that the *code
representation* ``CR^p`` - literal ``CREATE TABLE`` DDL including primary and
foreign keys, plus a few example rows - is the strongest question
representation for code-pretrained LLMs. ``render_schema`` produces that, and
``prune`` implements the CHESS Schema-Selector / DIN-SQL schema-linking
narrowing on top of it.
"""
from __future__ import annotations

import functools
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Column:
    name: str
    type: str
    pk: bool = False
    notnull: bool = False
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: List[Column] = field(default_factory=list)

    def column(self, name: str) -> Optional[Column]:
        low = name.lower()
        for c in self.columns:
            if c.name.lower() == low:
                return c
        return None


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class DBSchema:
    db_id: str
    path: str
    tables: List[Table] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)

    # -- lookups -----------------------------------------------------------
    def table(self, name: str) -> Optional[Table]:
        low = name.lower()
        for t in self.tables:
            if t.name.lower() == low:
                return t
        return None

    @property
    def n_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    def column_pairs(self) -> List[Tuple[str, str]]:
        return [(t.name, c.name) for t in self.tables for c in t.columns]

    # -- pruning (CHESS "Schema Selector" / DIN-SQL "schema linking") -------
    def prune(self, links: Dict[str, Sequence[str]]) -> "DBSchema":
        """Return a sub-schema containing only the selected tables/columns.

        ``links`` maps table name -> list of column names. Primary keys and
        any foreign key that connects two surviving tables are always kept, so
        the pruned schema stays joinable (a failure mode DIN-SQL section 3
        calls out explicitly).
        """
        wanted = {k.lower(): {c.lower() for c in v} for k, v in links.items() if v is not None}
        if not wanted:
            return self

        kept_tables: List[Table] = []
        for t in self.tables:
            if t.name.lower() not in wanted:
                continue
            sel = wanted[t.name.lower()]
            cols = [c for c in t.columns if c.pk or not sel or c.name.lower() in sel]
            if not cols:
                cols = list(t.columns)
            kept_tables.append(Table(name=t.name, columns=cols))

        if not kept_tables:
            return self

        names = {t.name.lower() for t in kept_tables}
        kept_fks = [
            fk for fk in self.foreign_keys
            if fk.from_table.lower() in names and fk.to_table.lower() in names
        ]
        # make sure join keys survived the column filter
        by_name = {t.name.lower(): t for t in kept_tables}
        for fk in kept_fks:
            for tname, cname in ((fk.from_table, fk.from_column), (fk.to_table, fk.to_column)):
                tbl = by_name[tname.lower()]
                if tbl.column(cname) is None:
                    src = self.table(tname)
                    orig = src.column(cname) if src else None
                    if orig is not None:
                        tbl.columns.append(orig)
        return DBSchema(db_id=self.db_id, path=self.path, tables=kept_tables, foreign_keys=kept_fks)


# --------------------------------------------------------------------------
# introspection
# --------------------------------------------------------------------------
def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def load_schema(
    db_path: str,
    db_id: Optional[str] = None,
    n_examples: int = 3,
    max_example_chars: int = 40,
) -> DBSchema:
    """Read tables, columns, PKs, FKs and a few sample values out of a SQLite file."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    db_id = db_id or os.path.splitext(os.path.basename(db_path))[0]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        table_names = [r[0] for r in cur.fetchall()]

        tables: List[Table] = []
        fks: List[ForeignKey] = []
        for tname in table_names:
            cols: List[Column] = []
            cur.execute(f"PRAGMA table_info({_quote(tname)})")
            for _cid, cname, ctype, notnull, _dflt, pk in cur.fetchall():
                col = Column(
                    name=cname,
                    type=(ctype or "TEXT").upper(),
                    pk=bool(pk),
                    notnull=bool(notnull),
                )
                if n_examples > 0:
                    col.examples = _sample_values(
                        cur, tname, cname, n_examples, max_example_chars
                    )
                cols.append(col)
            tables.append(Table(name=tname, columns=cols))

            cur.execute(f"PRAGMA foreign_key_list({_quote(tname)})")
            for row in cur.fetchall():
                # (id, seq, table, from, to, on_update, on_delete, match)
                to_table, from_col, to_col = row[2], row[3], row[4]
                if to_col is None:  # implicit reference to the target PK
                    tgt = next((t for t in tables if t.name.lower() == str(to_table).lower()), None)
                    if tgt is None:
                        continue
                    pk_cols = [c.name for c in tgt.columns if c.pk]
                    to_col = pk_cols[0] if pk_cols else None
                    if to_col is None:
                        continue
                fks.append(ForeignKey(tname, from_col, to_table, to_col))
        return DBSchema(db_id=db_id, path=db_path, tables=tables, foreign_keys=fks)
    finally:
        conn.close()


def _sample_values(cur, table: str, column: str, n: int, max_chars: int) -> List[str]:
    try:
        cur.execute(
            f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} "
            f"WHERE {_quote(column)} IS NOT NULL LIMIT {int(n)}"
        )
        out = []
        for (v,) in cur.fetchall():
            s = str(v)
            if len(s) > max_chars:
                s = s[: max_chars - 3] + "..."
            out.append(s)
        return out
    except sqlite3.Error:
        return []


@functools.lru_cache(maxsize=64)
def load_schema_cached(db_path: str, db_id: Optional[str] = None) -> DBSchema:
    return load_schema(db_path, db_id)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def render_schema(
    schema: DBSchema,
    style: str = "code",
    with_examples: bool = True,
    with_descriptions: bool = True,
    max_columns_per_table: Optional[int] = None,
) -> str:
    """Render a schema for a prompt.

    ``style="code"``   -> DAIL-SQL ``CR^p``: CREATE TABLE DDL + PK/FK.
    ``style="compact"``-> ``table: col1, col2, ...`` (cheap, for long schemas).
    """
    if style == "compact":
        lines = []
        for t in schema.tables:
            cols = t.columns[:max_columns_per_table] if max_columns_per_table else t.columns
            lines.append(f"{t.name}: " + ", ".join(c.name for c in cols))
        for fk in schema.foreign_keys:
            lines.append(f"-- FK {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
        return "\n".join(lines)

    out: List[str] = []
    for t in schema.tables:
        cols = t.columns[:max_columns_per_table] if max_columns_per_table else t.columns
        out.append(f"CREATE TABLE {t.name} (")
        body: List[str] = []
        for c in cols:
            line = f"  {c.name} {c.type}"
            if c.pk:
                line += " PRIMARY KEY"
            comment_bits: List[str] = []
            if with_descriptions and c.description:
                comment_bits.append(c.description)
            if with_examples and c.examples:
                comment_bits.append("examples: " + ", ".join(repr(e) for e in c.examples))
            if comment_bits:
                line += "  -- " + " | ".join(comment_bits)
            body.append(line)
        for fk in schema.foreign_keys:
            if fk.from_table.lower() == t.name.lower():
                body.append(
                    f"  FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})"
                )
        out.append(",\n".join(body))
        out.append(");")
    return "\n".join(out)


def schema_keywords(schema: DBSchema) -> List[str]:
    """Every identifier a schema-linking step could plausibly mention."""
    kws = set()
    for t in schema.tables:
        kws.add(t.name)
        for c in t.columns:
            kws.add(c.name)
    return sorted(kws)
