"""DuckDB warehouse: the analytical SQL lives in versioned .sql files.

src/warehouse.py only orchestrates: load the interim parquet into DuckDB,
execute every sql/*.sql in order, export the resulting tables to parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.data import ROOT, load_config

SQL_DIR = ROOT / "sql"


def _fwd(path: Path) -> str:
    # DuckDB accepts forward slashes on Windows; avoids escaping backslashes in SQL.
    return str(path).replace("\\", "/")


def build_warehouse(cfg: dict) -> dict:
    db_path = ROOT / "data" / cfg["warehouse"]["db_file"]
    interim = ROOT / cfg["data"]["interim_dir"] / cfg["data"]["interim_file"]
    processed = ROOT / cfg["data"]["processed_dir"]

    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            "CREATE OR REPLACE TABLE transactions AS "
            f"SELECT * FROM read_parquet('{_fwd(interim)}')"
        )
        for sql_file in sorted(SQL_DIR.glob("*.sql")):
            con.execute(sql_file.read_text(encoding="utf-8"))
            print(f"ran {sql_file.name}")

        processed.mkdir(parents=True, exist_ok=True)
        weekly_path = processed / cfg["warehouse"]["weekly_file"]
        abc_path = processed / cfg["warehouse"]["abc_file"]
        con.execute(f"COPY weekly_demand TO '{_fwd(weekly_path)}' (FORMAT PARQUET)")
        con.execute(f"COPY abc_classes TO '{_fwd(abc_path)}' (FORMAT PARQUET)")

        summary = {
            "n_skus": con.execute("SELECT COUNT(*) FROM abc_classes").fetchone()[0],
            "n_weeks": con.execute(
                "SELECT COUNT(DISTINCT week_start) FROM weekly_demand"
            ).fetchone()[0],
            "week_min": str(
                con.execute("SELECT MIN(week_start) FROM weekly_demand").fetchone()[0]
            ),
            "week_max": str(
                con.execute("SELECT MAX(week_start) FROM weekly_demand").fetchone()[0]
            ),
            "class_counts": con.execute(
                "SELECT abc_class, COUNT(*) FROM abc_classes GROUP BY 1 ORDER BY 1"
            ).fetchall(),
        }
        top = con.execute(
            "SELECT stock_code, total_units, total_revenue, abc_class "
            "FROM abc_classes ORDER BY revenue_rank LIMIT 5"
        ).fetchall()
    finally:
        con.close()

    print(
        f"SKUs: {summary['n_skus']:,}  weeks: {summary['n_weeks']}  "
        f"span: {summary['week_min']} -> {summary['week_max']}"
    )
    print(f"ABC class sizes: {summary['class_counts']}")
    print("Top SKUs by revenue:")
    for code, units, revenue, cls in top:
        print(f"  {code}  class {cls}  {units:,} units  £{revenue:,.0f}")
    return summary


def main() -> None:
    build_warehouse(load_config())


if __name__ == "__main__":
    main()
