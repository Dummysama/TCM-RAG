from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

CLEAN_DIR = Path("outputs/cleaned")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "knowledge_chunks.jsonl"


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _split_multi_value(text: str) -> List[str]:
    """
    将中文多值字段粗略拆分成列表（可后续再精细化）。
    常见分隔符：、,，;；/｜
    """
    t = _safe_str(text)
    if not t:
        return []
    for sep in ["；", ";", "，", ",", "、", "/", "｜", "|", "\n", "\t"]:
        t = t.replace(sep, "、")
    parts = [p.strip() for p in t.split("、") if p.strip()]
    # 去重但保持顺序
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_herb_chunks(df_herb: pd.DataFrame) -> List[Dict[str, Any]]:
    chunks = []
    for _, row in df_herb.iterrows():
        herb_id = _safe_str(row.get("TCM_HerbID"))
        name = _safe_str(row.get("药材") or row.get("药材中文名") or row.get("药材名称") or row.get("Herb"))
        pinyin = _safe_str(row.get("PINYIN"))
        latin = _safe_str(row.get("LATIN"))
        english = _safe_str(row.get("English_Name"))
        typ = _safe_str(row.get("TYPE"))
        prop = _safe_str(row.get("Property_CHN"))
        flavor = _safe_str(row.get("Flavor_CHN"))
        meridian = _safe_str(row.get("Meridian_Tropism_CHN"))
        indication = _safe_str(row.get("Indication_CHN"))
        efficacy = _safe_str(row.get("功效"))
        toxicity = _safe_str(row.get("毒性"))
        alias = _safe_str(row.get("别名"))

        # 规范一些可用于检索的字段（列表化）
        meridian_list = _split_multi_value(meridian)
        efficacy_list = _split_multi_value(efficacy)
        indication_list = _split_multi_value(indication)

        text_lines = [
            f"中药名称：{name}" if name else f"中药ID：{herb_id}",
            f"拼音：{pinyin}" if pinyin else "",
            f"拉丁名：{latin}" if latin else "",
            f"英文名：{english}" if english else "",
            f"来源类型：{typ}" if typ else "",
            f"性：{prop}" if prop else "",
            f"味：{flavor}" if flavor else "",
            f"归经：{'、'.join(meridian_list)}" if meridian_list else (f"归经：{meridian}" if meridian else ""),
            f"功效：{'、'.join(efficacy_list)}" if efficacy_list else (f"功效：{efficacy}" if efficacy else ""),
            f"主治/适应证：{'、'.join(indication_list)}" if indication_list else (f"主治/适应证：{indication}" if indication else ""),
            f"别名：{alias}" if alias else "",
            f"毒性：{toxicity}" if toxicity else "",
        ]
        text = "\n".join([x for x in text_lines if x])

        chunks.append({
            "id": f"herb::{herb_id}" if herb_id else f"herb::{name}",
            "type": "herb",
            "text": text,
            "metadata": {
                "TCM_HerbID": herb_id,
                "name": name,
                "pinyin": pinyin,
                "source_table": "herb",
                "source_file": "中药.xlsx",
            }
        })
    return chunks


def build_prescription_chunks(df_rx: pd.DataFrame) -> List[Dict[str, Any]]:
    chunks = []
    for _, row in df_rx.iterrows():
        rx_id = _safe_str(row.get("TCM_PrescriptionID"))
        name = _safe_str(row.get("方剂名") or row.get("方剂中文名"))
        pinyin = _safe_str(row.get("方剂拼音名"))
        symptom = _safe_str(row.get("主治症状（功效）") or row.get("主治症状"))
        syndrome = _safe_str(row.get("主治证候"))
        source = _safe_str(row.get("来源"))
        symptom_id = _safe_str(row.get("TCM_SymptomID"))
        syndrome_id = _safe_str(row.get("TCM_SyndromeID"))

        text_lines = [
            f"方剂名称：{name}" if name else f"方剂ID：{rx_id}",
            f"拼音：{pinyin}" if pinyin else "",
            f"主治症状/功效：{symptom}" if symptom else "",
            f"主治证候：{syndrome}" if syndrome else "",
            f"出处：{source}" if source else "",
        ]
        text = "\n".join([x for x in text_lines if x])

        chunks.append({
            "id": f"prescription::{rx_id}" if rx_id else f"prescription::{name}",
            "type": "prescription",
            "text": text,
            "metadata": {
                "TCM_PrescriptionID": rx_id,
                "name": name,
                "pinyin": pinyin,
                "TCM_SymptomID": symptom_id,
                "TCM_SyndromeID": syndrome_id,
                "source_table": "prescription",
                "source_file": "TCM方剂（含中成药）.xlsx",
            }
        })
    return chunks


def build_mechanism_chunks(
    df_hc: pd.DataFrame,
    df_ct: pd.DataFrame,
    df_herb: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    生成“中药->成分->靶点”的机制型 chunks：
    - 以 herb_id 为单位聚合
    - 每味药保留最多 max_compounds 个成分，每个成分保留最多 max_targets 个靶点
    这样能控制 chunk 长度，便于后续 embedding 与检索。
    """
    max_compounds = 30
    max_targets = 10

    # 建立 inchikey -> targets 映射
    df_ct2 = df_ct.copy()
    # 列名统一去空格，避免奇怪 KeyError
    df_ct2.columns = [str(c).strip() for c in df_ct2.columns]
    inch_col = "Inchikey"
    gene_col = "Gene Entrez ID"
    sym_col = "Targets (Coding gene offical symbol)"

    # 有些行可能 gene/symbol 缺失，先丢掉
    df_ct2 = df_ct2.dropna(subset=[inch_col, gene_col])

    # 归一化
    df_ct2[inch_col] = df_ct2[inch_col].astype(str).str.strip()
    df_ct2[gene_col] = df_ct2[gene_col].astype(str).str.strip()
    if sym_col in df_ct2.columns:
        df_ct2[sym_col] = df_ct2[sym_col].astype(str).str.strip()

    targets_map: Dict[str, List[Dict[str, str]]] = {}
    for inchikey, sub in df_ct2.groupby(inch_col):
        items = []
        # 去重：同一 inchikey + entrez 只保留一条
        sub2 = sub.drop_duplicates(subset=[gene_col])
        for _, r in sub2.iterrows():
            items.append({
                "entrez_id": _safe_str(r.get(gene_col)),
                "symbol": _safe_str(r.get(sym_col)) if sym_col in sub2.columns else ""
            })
        targets_map[inchikey] = items

    # 建立 herb_id -> compounds 映射
    df_hc2 = df_hc.copy()
    df_hc2.columns = [str(c).strip() for c in df_hc2.columns]
    df_hc2 = df_hc2.dropna(subset=["TCM_HerbID", "Inchikey"])
    df_hc2["TCM_HerbID"] = df_hc2["TCM_HerbID"].astype(str).str.strip()
    df_hc2["Inchikey"] = df_hc2["Inchikey"].astype(str).str.strip()

    herb_name_map = {}
    if df_herb is not None and "TCM_HerbID" in df_herb.columns:
        tmp = df_herb[["TCM_HerbID", "药材"]].copy()
        tmp["TCM_HerbID"] = tmp["TCM_HerbID"].astype(str).str.strip()
        tmp["药材"] = tmp["药材"].astype(str).str.strip()
        herb_name_map = dict(zip(tmp["TCM_HerbID"], tmp["药材"]))

    chunks = []
    for herb_id, sub in df_hc2.groupby("TCM_HerbID"):
        herb_name = herb_name_map.get(herb_id, "")
        sub2 = sub.drop_duplicates(subset=["Inchikey"]).head(max_compounds)

        lines = []
        header = f"中药：{herb_name}（{herb_id}）" if herb_name else f"中药ID：{herb_id}"
        lines.append(header)
        lines.append("关联成分与靶点（部分展示）：")

        mech_items = []
        for _, r in sub2.iterrows():
            inchikey = _safe_str(r.get("Inchikey"))
            pubchem = _safe_str(r.get("Pubchem_CID"))
            smiles = _safe_str(r.get("Canonical smiles"))
            targets = targets_map.get(inchikey, [])[:max_targets]

            t_str = "；".join(
                [f"{t['symbol']}({t['entrez_id']})" if t["symbol"] else f"{t['entrez_id']}" for t in targets]
            ) if targets else "未关联到靶点"

            lines.append(f"- 成分InChIKey：{inchikey} | PubChem:{pubchem or 'NA'} | 靶点：{t_str}")

            mech_items.append({
                "inchikey": inchikey,
                "pubchem_cid": pubchem,
                "smiles": smiles,
                "targets": targets,
            })

        text = "\n".join(lines)

        chunks.append({
            "id": f"mechanism::{herb_id}",
            "type": "mechanism",
            "text": text,
            "metadata": {
                "TCM_HerbID": herb_id,
                "herb_name": herb_name,
                "max_compounds": max_compounds,
                "max_targets": max_targets,
                "source_table": "herb_compound+compound_target",
                "source_files": ["中药-化合物关联.xlsx", "中药化合物-靶标关联.xlsx"],
                "mechanism_items": mech_items,  # 供后续证据展示或调试
            }
        })
    return chunks


def main() -> None:
    # 读取清洗后的 Parquet
    herb_path = CLEAN_DIR / "herb.parquet"
    rx_path = CLEAN_DIR / "prescription.parquet"
    hc_path = CLEAN_DIR / "herb_compound.parquet"
    ct_path = CLEAN_DIR / "compound_target.parquet"

    for p in [herb_path, rx_path, hc_path, ct_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing cleaned file: {p}")

    df_herb = pd.read_parquet(herb_path)
    df_rx = pd.read_parquet(rx_path)
    df_hc = pd.read_parquet(hc_path)
    df_ct = pd.read_parquet(ct_path)

    # 生成 chunks
    chunks: List[Dict[str, Any]] = []
    chunks.extend(build_herb_chunks(df_herb))
    chunks.extend(build_prescription_chunks(df_rx))
    chunks.extend(build_mechanism_chunks(df_hc, df_ct, df_herb=df_herb))

    # 保存 jsonl
    _write_jsonl(OUT_JSONL, chunks)

    print(f"Saved chunks: {OUT_JSONL}  (count={len(chunks)})")
    print("Example:")
    print(json.dumps(chunks[0], ensure_ascii=False, indent=2)[:1000])


if __name__ == "__main__":
    main()