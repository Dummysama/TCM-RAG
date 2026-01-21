from pathlib import Path
import pandas as pd

# =========================
# Paths
# =========================
DATA_DIR = Path("data")
OUT_DIR = Path("outputs/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Core tables only (重点数据)
# =========================
TABLE_CONFIG = {
    "herb": {
        "file": "中药.xlsx",
        "pk": ["TCM_HerbID"],
    },
    "prescription": {
        "file": "TCM方剂（含中成药）.xlsx",
        "pk": ["TCM_PrescriptionID"],
    },
    "herb_compound": {
        "file": "中药-化合物关联.xlsx",
        "pk": ["TCM_HerbID", "Inchikey"],
    },
    "compound_target": {
        "file": "中药化合物-靶标关联.xlsx",
        "pk": ["Inchikey", "Gene Entrez ID"],
    },
}

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """仅做最小必要规范化：去掉列名首尾空格，避免 KeyError。"""
    df.columns = [str(c).strip() for c in df.columns]
    return df

def clean_table(name: str, cfg: dict) -> None:
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"[{name}] file not found: {path}")

    df = pd.read_excel(path)
    df = _strip_columns(df)

    pk = cfg["pk"]
    missing_pk_cols = [c for c in pk if c not in df.columns]
    if missing_pk_cols:
        raise KeyError(f"[{name}] missing PK columns in sheet: {missing_pk_cols}")

    original_rows = len(df)

    # 1) 删除主键缺失行（关系表尤其关键）
    df = df.dropna(subset=pk)

    # 2) 删除主键重复行（保留第一条）
    df = df.drop_duplicates(subset=pk, keep="first")

    cleaned_rows = len(df)
    removed = original_rows - cleaned_rows
    print(f"[{name}] {original_rows} → {cleaned_rows} (removed {removed})")

    # 3) 保存为 parquet（工程推荐的中间格式）
    out_path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)

def main() -> None:
    for name, cfg in TABLE_CONFIG.items():
        clean_table(name, cfg)

    print("\nAll cleaned core tables saved to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()