from __future__ import annotations

import ast
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "RELEASE_MANIFEST.csv",
    "config/path_config.example.json",
    "docs/DATA_ACCESS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/WORKFLOW_ORDER.md",
    "docs/MANUSCRIPT_OUTPUT_MAP.md",
    "docs/INPUT_REQUIREMENTS.md",
    "docs/OUTPUT_DICTIONARY.md",
    "docs/PRIVACY_AND_SECURITY.md",
    "docs/PUBLIC_RELEASE_AUDIT.md",
    "data/README.md",
]

FINAL_SCRIPTS = [
    "scripts/sensitivity/01_strict_label_sensitivity_v2.py",
    "scripts/sensitivity/02_generate_discrimination_calibration_v2.py",
    "scripts/sensitivity/03_generate_paired_model_comparison_v2.py",
    "scripts/sensitivity/04_generate_subgroup_robustness_v2.py",
    "scripts/sensitivity/05_generate_calibration_uncertainty_v2.py",
    "scripts/sensitivity/06_generate_submission_figures_tables_v2.py",
    "scripts/sensitivity/08_check_submission_enhancement_v2.py",
    "scripts/sensitivity/09_check_v9_final_consistency.py",
]

ALLOWED_AGGREGATE_CSV = {
    "example_outputs/aggregate_tables/strict_label_flow_by_wave.csv",
    "example_outputs/aggregate_tables/strict_label_metrics_with_95CI.csv",
    "example_outputs/aggregate_tables/primary_vs_strict_label_comparison.csv",
    "example_outputs/aggregate_tables/paired_model_differences_with_95CI.csv",
    "example_outputs/aggregate_tables/calibration_uncertainty_summary.csv",
    "example_outputs/aggregate_tables/subgroup_performance_with_95CI.csv",
    "example_outputs/aggregate_tables/subgroup_sample_adequacy.csv",
    "example_outputs/aggregate_tables/Figure2_ROC_curve_data.csv",
    "example_outputs/aggregate_tables/Figure2_PR_curve_data.csv",
    "example_outputs/aggregate_tables/Figure2_calibration_bin_data.csv",
    "example_outputs/aggregate_tables/Figure2_calibration_summary.csv",
    "example_outputs/aggregate_tables/Figure4_plot_data.csv",
    "example_outputs/aggregate_tables/Table2_main_model_validation_concise.csv",
    "example_outputs/aggregate_tables/SuppTable_wave1_oof_model_performance.csv",
    "example_outputs/aggregate_tables/SuppTable_strict_label_flow.csv",
    "example_outputs/aggregate_tables/SuppTable_strict_label_model_performance.csv",
    "example_outputs/aggregate_tables/SuppTable_primary_vs_strict_label.csv",
    "example_outputs/aggregate_tables/SuppTable_paired_model_differences.csv",
    "example_outputs/aggregate_tables/SuppTable_calibration_uncertainty.csv",
    "example_outputs/aggregate_tables/SuppTable_subgroup_performance.csv",
    "example_outputs/aggregate_tables/SuppFigureS4_original_vs_id_isolated_source_data.csv",
    "example_outputs/aggregate_tables/SuppFigureS6_learning_curve_source_data.csv",
    "RELEASE_MANIFEST.csv",
}

BANNED_SUFFIXES = {".dta", ".parquet", ".feather", ".pkl", ".pickle", ".joblib", ".sav", ".key", ".pem"}
RAW_SUFFIXES = {".sas7bdat", ".xpt", ".rds", ".rda"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".yml", ".yaml", ".cff", ".csv", ".gitignore", ".sha256"}
PERSONAL_PATH_MARKERS = ["/" + "Users" + "/", "/" + "home" + "/"]
WINDOWS_USER_PATH_RE = re.compile(r"C:" + r"\\Users\\")
SANITISED_HOME_PLACEHOLDER = "<USER" + "_HOME>"
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}", re.I),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"][^'\"]{6,}['\"]"),
    re.compile(r"-----BEGIN (RSA |OPENSSH |DSA |EC |PGP )?PRIVATE KEY-----"),
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def add(checks, ok: bool, message: str, critical: bool = True):
    checks.append(("PASS" if ok else ("FAIL" if critical else "WARNING"), message, critical))


def text_for(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    checks = []
    add(
        checks,
        ROOT.name in {"github_release_v1", "charls_possible_sarcopenia_screening_v1.0.0"},
        "Checker root has an expected release name",
        True,
    )

    files = [p for p in ROOT.rglob("*") if p.is_file()]
    dirs = [p for p in ROOT.rglob("*") if p.is_dir()]

    add(checks, not (ROOT / ".git").exists(), "No .git directory")
    for d in dirs:
        name = d.name
        if name in {".git", "__pycache__", ".ipynb_checkpoints"}:
            add(checks, False, f"Banned directory present: {rel(d)}")
    for p in files:
        if p.name in {".DS_Store"}:
            add(checks, False, f"Banned OS/cache file present: {rel(p)}")
        if p.suffix in {".pyc", ".pyo", ".pyd"}:
            add(checks, False, f"Compiled Python file present: {rel(p)}")
        if p.suffix.lower() in BANNED_SUFFIXES | RAW_SUFFIXES:
            add(checks, False, f"Banned raw/model/secret-like extension present: {rel(p)}")
        if p.stat().st_size > 10 * 1024 * 1024:
            add(checks, False, f"File larger than 10 MB: {rel(p)}")
        low = rel(p).lower()
        if p.suffix.lower() == ".docx":
            add(checks, False, f"Manuscript/document draft included: {rel(p)}")
        if p.suffix.lower() == ".csv" and rel(p) not in ALLOWED_AGGREGATE_CSV:
            add(checks, False, f"Unexpected CSV outside aggregate allowlist: {rel(p)}")
        if p.suffix.lower() in {".csv", ".tsv"} and any(term in low for term in ["prediction", "model_input", "participant_id", "row_level"]):
            add(checks, False, f"Potential participant-level data file included: {rel(p)}")

    for req in REQUIRED_FILES:
        add(checks, (ROOT / req).exists(), f"Required file exists: {req}")
    for req in FINAL_SCRIPTS:
        add(checks, (ROOT / req).exists(), f"Final v2 workflow script present: {req}")

    # Text scans.
    for p in files:
        if p.name == "CHECKSUMS.sha256":
            continue
        if p.suffix.lower() in TEXT_SUFFIXES or p.name in {".gitignore", "LICENSE"}:
            txt = text_for(p)
            if any(marker in txt for marker in PERSONAL_PATH_MARKERS) or WINDOWS_USER_PATH_RE.search(txt):
                add(checks, False, f"Absolute personal path found in {rel(p)}")
            if SANITISED_HOME_PLACEHOLDER in txt and p.suffix == ".py":
                add(checks, False, f"Sanitised placeholder path remains in executable script: {rel(p)}")
            for pat in SECRET_PATTERNS:
                if pat.search(txt):
                    add(checks, False, f"Potential secret pattern in {rel(p)}")
            if "CHARLS data are not distributed" in txt or "not include CHARLS participant-level data" in txt:
                pass

    readme = text_for(ROOT / "README.md")
    old_manuscript_title = (
        "Model complexity"
        " and practical utility"
        " in screening for possible sarcopenia among older Chinese adults:"
        " development and stricter"
        " ID-isolated cross-wave evaluation using CHARLS"
    )
    add(checks, "CHARLS participant-level microdata are not included" in readme, "README documents CHARLS data restrictions")
    add(checks, "not fully independent external validation" in readme, "README avoids overstating external validation")
    add(checks, "Yu Yue" in readme and "Chunhua Zhang" in readme, "README includes confirmed authors")
    add(checks, "zch20080808@126.com" in readme, "README includes confirmed corresponding-author email")
    add(checks, "Does cross-wave evaluation overestimate screening performance? An ID-isolated comparison of logistic regression and XGBoost for possible sarcopenia in CHARLS" in readme, "README includes current manuscript title")
    add(checks, old_manuscript_title not in readme, "README excludes superseded complete manuscript title")
    add(checks, old_manuscript_title not in text_for(ROOT / "CITATION.cff"), "CITATION excludes superseded complete manuscript title")
    add(checks, (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"), "MIT license exists")
    add(checks, "Code for participant-isolated evaluation of possible sarcopenia screening models in CHARLS" in text_for(ROOT / "CITATION.cff"), "CITATION.cff exists and has expected repository title")

    # Python syntax and obvious relative helper checks.
    py_files = [p for p in files if p.suffix == ".py"]
    for p in py_files:
        try:
            ast.parse(text_for(p), filename=str(p))
        except SyntaxError as exc:
            add(checks, False, f"Python syntax error in {rel(p)}: {exc}")
    add(checks, (ROOT / "scripts/sensitivity/_utils_v2.py").exists(), "v2 helper `_utils_v2.py` is present")
    for p in (ROOT / "scripts/sensitivity").glob("*_v2.py"):
        if "_utils_v2" in text_for(p):
            add(checks, (p.parent / "_utils_v2.py").exists(), f"Relative helper import resolvable for {rel(p)}")

    manifest_path = ROOT / "RELEASE_MANIFEST.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        manifest_files = {r["release_path"] for r in rows}
        actual_files = {rel(p) for p in files}
        missing = sorted(actual_files - manifest_files)
        extra = sorted(manifest_files - actual_files)
        add(checks, not missing, f"Manifest includes every release file; missing={missing[:5]}")
        add(checks, not extra, f"Manifest has no nonexistent files; extra={extra[:5]}")
        unsafe = [r["release_path"] for r in rows if r.get("public_safe") not in {"yes", "yes_pending_author_review"}]
        add(checks, not unsafe, f"Manifest marks included files public-safe; unsafe={unsafe[:5]}")

    checksums = ROOT / "CHECKSUMS.sha256"
    if checksums.exists():
        listed = {}
        for line in checksums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, path = line.split("  ", 1)
            listed[path] = digest
        checksum_targets = {rel(p) for p in files if p.name != "CHECKSUMS.sha256"}
        add(checks, set(listed) == checksum_targets, "CHECKSUMS.sha256 covers every release file except itself")
        bad = []
        for path, digest in listed.items():
            h = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            if h != digest:
                bad.append(path)
        add(checks, not bad, f"Checksum digests match current files; bad={bad[:5]}")
    else:
        add(checks, False, "CHECKSUMS.sha256 exists", critical=False)

    # Default workflow should not depend only on superseded scripts.
    map_text = text_for(ROOT / "docs/MANUSCRIPT_OUTPUT_MAP.md")
    add(checks, "02_generate_discrimination_calibration_v2.py" in map_text, "Figure 2 mapped to v2 script")
    add(checks, "03_generate_paired_model_comparison_v2.py" in map_text, "Figure 4 mapped to v2 script")
    add(checks, "24_generate_figure1_participant_flow.py" in map_text and "Final grayscale workflow" in map_text, "Figure 1 mapped to final grayscale workflow")

    critical_failed = any(status == "FAIL" and critical for status, _, critical in checks)
    warnings = any(status == "WARNING" for status, _, _ in checks)
    overall = "FAIL" if critical_failed else ("PASS WITH WARNINGS" if warnings else "PASS")

    lines = ["# Public Release Audit", "", f"Overall status: **{overall}**", "", "| Status | Check | Critical |", "|---|---|---:|"]
    for status, msg, critical in checks:
        lines.append(f"| {status} | {msg} | {critical} |")
    (ROOT / "docs/PUBLIC_RELEASE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(overall)
    if critical_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
