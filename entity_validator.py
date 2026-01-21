# 实体存在性校验
# 06_entity_validator.py
import json
from pathlib import Path

META_PATH = Path("outputs/index/meta.jsonl")

def load_all_entities():
    herbs = set()
    prescriptions = set()

    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            t = it.get("type")
            name = (it.get("metadata") or {}).get("name")
            if not name:
                continue
            if t == "herb":
                herbs.add(name)
            elif t == "prescription":
                prescriptions.add(name)

    return herbs, prescriptions


HERBS, PRESCRIPTIONS = load_all_entities()


def validate_entity(entity: str, intent: str) -> bool:
    if not entity:
        return False

    if intent.startswith("HERB"):
        return entity in HERBS

    if intent.startswith("PRESCRIPTION"):
        return entity in PRESCRIPTIONS

    return False