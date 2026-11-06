"""维护者工具：准备 Numeral / Gravity / Equation 官方课程数据。

学生直接读取已分发的 real_cleaned.csv，无需处理原始 CSV。
本工具只执行官方数据核验、分类、筛选和 Numeral 映射检查；
80/10/10 划分、跨集合泄漏审计与 Baseline 留在 00 Notebook。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TASK_ORDER = ("numeral_system", "gravity", "equation_numeric")
TARGET_TASKS = set(TASK_ORDER)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, payload):
    path = Path(path)
    data = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"拒绝覆盖既有版本产物: {path}")
    if not path.exists():
        path.write_bytes(data)


TEMPLATE_PREFIXES = [
 ("gravity", "In Alice's Wonderland, the gravitational constant has been secretly changed."),
 ("bit_manipulation", "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."),
 ("equation", "In Alice's Wonderland, a secret set of transformation rules is applied to equations."),
 ("unit_conversion", "In Alice's Wonderland, a secret unit conversion is applied to measurements."),
 ("encryption", "In Alice's Wonderland, secret encryption rules are used on text."),
 ("numeral_system", "In Alice's Wonderland, numbers are secretly converted into a different numeral system."),
]

NUMERIC_QUERY = re.compile(r"Now, determine the result for:\s*(\d{2})([^\d\s])(\d{2})\s*$")

NUMERIC_EXAMPLE = re.compile(r"^\d{2}([^\d\s])\d{2}\s*=\s*.+$")

def classify_prompt(prompt):
    for task, prefix in TEMPLATE_PREFIXES:
        if not prompt.startswith(prefix): continue
        if task != "equation": return {"task_type": task, "classification_rule": "exact_template_prefix", "equation_subtype": ""}
        examples = [line.strip() for line in prompt.splitlines() if " = " in line]
        query = NUMERIC_QUERY.search(prompt)
        if query and examples and all(NUMERIC_EXAMPLE.fullmatch(line) for line in examples):
            operators = {NUMERIC_EXAMPLE.fullmatch(line).group(1) for line in examples}
            return {"task_type":"equation_numeric", "classification_rule":"exact_prefix_and_all_numeric_examples_and_query",
                    "equation_subtype":"deduce" if query.group(2) in operators else "guess"}
        if not re.search(r"\d", prompt):
            return {"task_type":"equation_symbolic", "classification_rule":"exact_equation_prefix_no_digits", "equation_subtype":"symbolic"}
        return {"task_type":"unknown", "classification_rule":"equation_numeric_structure_unconfirmed", "equation_subtype":"undetermined"}
    return {"task_type":"unknown", "classification_rule":"unregistered_template", "equation_subtype":""}


ROMAN_GRAMMAR = re.compile(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
ROMAN_VALUES = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

def arabic_to_roman(number):
    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= 3999: raise ValueError("标准范围为 1..3999")
    thousands = ("", "M", "MM", "MMM")
    hundreds = ("", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM")
    tens = ("", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC")
    ones = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX")
    return thousands[number//1000] + hundreds[number//100%10] + tens[number//10%10] + ones[number%10]

def canonical_roman_to_int(roman):
    if not isinstance(roman, str) or not roman or not ROMAN_GRAMMAR.fullmatch(roman): raise ValueError("非规范罗马数字")
    values = [ROMAN_VALUES[char] for char in roman]
    result = sum(-value if index+1 < len(values) and value < values[index+1] else value for index, value in enumerate(values))
    if not 1 <= result <= 3999: raise ValueError("超出标准范围")
    return result

def verify_roman_number(number, roman):
    if not re.fullmatch(r"[0-9]+", str(number)): return False
    try: return canonical_roman_to_int(roman) == int(number)
    except (TypeError, ValueError, KeyError): return False

def parse_numeral_prompt(prompt):
    pairs = [(int(a), b) for a,b in re.findall(r"(?m)^(\d+)\s*->\s*([A-Z]+)\s*$", prompt)]
    mapping_lines = [line for line in prompt.splitlines() if "->" in line]
    query = re.search(r"Now, write the number (\d+) in the Wonderland numeral system\.\s*$", prompt)
    if not query or not pairs or len(mapping_lines) != len(pairs): raise ValueError("Numeral 题面解析失败")
    return pairs, int(query.group(1))


def prompt_hash(text): return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

def normalize_prompt(text): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).strip()).casefold()

def structure_fingerprint(text, task_type):
    text = normalize_prompt(text)
    if task_type == "numeral_system": text = re.sub(r"\b[ivxlcdm]+\b", "<roman>", text)
    text = re.sub(r"\d{2}" if task_type == "equation_numeric" else r"[-+]?\d+(?:\.\d+)?", "<num>", text)
    return prompt_hash(text)


def prepare_official_dataset(raw_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    if not {"id", "prompt", "answer"}.issubset(raw_df.columns): raise ValueError("原始 CSV 字段缺失")
    source_sha256 = sha256(raw_path)
    raw_profile = {"rows":len(raw_df), "columns":list(raw_df.columns), "dtypes":raw_df.dtypes.astype(str).to_dict(),
                   "missing":raw_df.isna().sum().to_dict(), "empty":raw_df.eq("").sum().to_dict(),
                   "duplicate_id_rows":int(raw_df.id.duplicated(keep=False).sum()),
                   "duplicate_prompt_rows":int(raw_df.prompt.duplicated(keep=False).sum()), "sha256":source_sha256}
    assert not raw_df[["id","prompt","answer"]].eq("").any().any()
    assert not raw_df.id.duplicated().any() and not raw_df.prompt.duplicated().any()
    classified = raw_df[["id","prompt","answer"]].join(pd.DataFrame(raw_df.prompt.map(classify_prompt).tolist()))
    classified.insert(0, "source_row", np.arange(2, len(classified) + 2))
    classified["source_dataset"] = "NVIDIA Nemotron Model Reasoning Challenge / train.csv"
    classified["source_sha256"] = source_sha256
    classified["kept_for_course"] = classified.task_type.isin(TARGET_TASKS)
    classified["exclusion_reason"] = np.where(classified.kept_for_course, "", "excluded_" + classified.task_type)
    cleaned_df = classified[classified.kept_for_course].copy().reset_index(drop=True)
    assert set(cleaned_df.task_type) == TARGET_TASKS
    write_once(output_dir / "classification_audit.csv", classified.to_csv(index=False))

    numeral_audit_rows = []
    for row in cleaned_df[cleaned_df.task_type == "numeral_system"].itertuples():
        pairs, query = parse_numeral_prompt(row.prompt)
        for index, (number, roman) in enumerate(pairs):
            numeral_audit_rows.append({"id":row.id,"kind":"example","index":index,"arabic":number,"roman":roman,
                                      "valid":verify_roman_number(number,roman),"encoder_agrees":arabic_to_roman(number)==roman})
        numeral_audit_rows.append({"id":row.id,"kind":"query_answer","index":-1,"arabic":query,"roman":row.answer,
                                  "valid":verify_roman_number(query,row.answer),"encoder_agrees":arabic_to_roman(query)==row.answer})
    numeral_audit = pd.DataFrame(numeral_audit_rows)
    assert numeral_audit.valid.all() and numeral_audit.encoder_agrees.all()
    write_once(output_dir / "numeral_mapping_audit.csv", numeral_audit.to_csv(index=False))

    cleaned_df["prompt_hash"] = cleaned_df.prompt.map(prompt_hash)
    cleaned_df["normalized_prompt_hash"] = cleaned_df.prompt.map(lambda text: prompt_hash(normalize_prompt(text)))
    cleaned_df["structure_fingerprint"] = [structure_fingerprint(row.prompt, row.task_type) for row in cleaned_df.itertuples()]
    write_once(output_dir / "real_cleaned.csv", cleaned_df.to_csv(index=False))
    report = {
        "schema_version": "nemotron-official-preparation/v1",
        "prepared_by": "maintainer",
        "tasks": list(TASK_ORDER),
        "source": {"path": raw_path.name, "sha256": source_sha256, "rows": len(raw_df)},
        "raw": raw_profile,
        "classification_counts": classified.task_type.value_counts().to_dict(),
        "unknown_rows": int(classified.task_type.eq("unknown").sum()),
        "cleaned_rows": len(cleaned_df),
        "cleaned_by_task": cleaned_df.task_type.value_counts().to_dict(),
        "numeral_mapping": {
            "example_count": int((numeral_audit.kind == "example").sum()),
            "query_count": int((numeral_audit.kind == "query_answer").sum()),
            "all_valid": bool(numeral_audit.valid.all() and numeral_audit.encoder_agrees.all()),
        },
        "files": {name: sha256(output_dir / name) for name in
                  ("real_cleaned.csv", "classification_audit.csv", "numeral_mapping_audit.csv")},
    }
    write_once(output_dir / "official_preparation_manifest.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "train.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=project_root / "data/processed/tasksets/roman_gravity_equation_v1")
    args = parser.parse_args()
    report = prepare_official_dataset(args.input, args.output_dir)
    print(json.dumps({key: report[key] for key in ("source", "cleaned_rows", "cleaned_by_task", "numeral_mapping")},
                     ensure_ascii=False, indent=2))
