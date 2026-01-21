from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

FILES = {
    "herb": DATA_DIR / "中药.xlsx",
    "prescription": DATA_DIR / "TCM方剂（含中成药）.xlsx",
    "herb_compound": DATA_DIR / "中药-化合物关联.xlsx",
    "compound_target": DATA_DIR / "中药化合物-靶标关联.xlsx",
    "symptom": DATA_DIR / "中医症状.xlsx",
    "syndrome": DATA_DIR / "中医证候.xlsx",
    "transcriptome": DATA_DIR / "中药转录组学数据 (2).xlsx",
}

# 每张表的主键（按你数据说明）
PRIMARY_KEYS = {
    "herb": ["TCM_HerbID"],
    "prescription": ["TCM_PrescriptionID"],
    "herb_compound": ["TCM_HerbID", "Inchikey"],
    "compound_target": ["Inchikey", "Gene Entrez ID"],
    "symptom": ["TCM_SymptomID"],
    "syndrome": ["TCM_SyndromeID"],
    "transcriptome": ["Inchikey", "Gene Entrez ID", "Cell line"],
}

def profile_one(name: str, path: Path) -> dict:
    if not path.exists():
        return {"table": name, "path": str(path), "exists": False}

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]  # 列名去空格

    info = {
        "table": name,
        "path": str(path),
        "exists": True,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }

    # 缺失概览
    na_rate = (df.isna().mean() * 100).round(2).to_dict()
    info["missing_rate_percent_top10"] = dict(sorted(na_rate.items(), key=lambda x: -x[1])[:10])

    # 主键质量
    pk = PRIMARY_KEYS.get(name)
    if pk:
        # 主键字段是否存在
        missing_cols = [c for c in pk if c not in df.columns]
        info["primary_key"] = pk
        info["pk_missing_columns"] = missing_cols

        if not missing_cols:
            pk_df = df[pk]
            info["pk_null_rows"] = int(pk_df.isna().any(axis=1).sum())
            info["pk_duplicate_rows"] = int(pk_df.duplicated().sum())
    else:
        info["primary_key"] = None

    return info

def main():
    results = []
    for name, path in FILES.items():
        results.append(profile_one(name, path))

    # 打印摘要
    print("=== Data Profile Summary ===")
    for r in results:
        if not r["exists"]:
            print(f"[MISSING] {r['table']}: {r['path']}")
            continue
        print(f"[OK] {r['table']}: rows={r['rows']}, cols={r['cols']}")
        if r.get("primary_key"):
            print(f"     PK={r['primary_key']} null_rows={r.get('pk_null_rows')} dup_rows={r.get('pk_duplicate_rows')}")
            if r.get("pk_missing_columns"):
                print(f"     !! PK columns missing: {r['pk_missing_columns']}")

    # 保存报告（给论文/记录用）
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "data_profile_report.json").write_text(
        pd.Series(results).to_json(force_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("\nSaved: outputs/data_profile_report.json")

if __name__ == "__main__":
    main()