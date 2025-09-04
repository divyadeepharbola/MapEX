#!/usr/bin/env python3
# PEP 8 Naming Checker (filenames, package folders, and class names)
#
# Usage:
#   python pep8_name_lint.py [PATH] [--output CSV] [--no-warn-underscore-packages] [--git-mv-plan]
# Examples:
#   python pep8_name_lint.py .
#   python pep8_name_lint.py D:\code\MapEX_100 --output naming_report.csv --git-mv-plan
#
# This tool checks:
#   • Module filenames (*.py): short, all-lowercase; underscores allowed; no hyphens; no capitals.
#   • Package/Folder names containing Python code: all-lowercase; underscores discouraged (warn by default).
#   • Class names inside .py files: CapWords/PascalCase (no underscores). Leading "_" allowed but discouraged (warn).

import argparse
import csv
import os
import re
import ast
from pathlib import Path
from typing import Dict

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "build", "dist", ".idea", ".vscode", ".tox",
    ".eggs", "node_modules"
}

# Modules: short, all-lowercase; underscores allowed; digits allowed.
MODULE_RE = re.compile(r"^_?[a-z][a-z0-9_]*\.py$")
SPECIAL_MODULES = {"__init__.py", "__main__.py", "conftest.py"}

# Packages (dirs with Python code): lowercase; underscores discouraged (warn).
PACKAGE_OK_RE = re.compile(r"^_?[a-z][a-z0-9_]*$")

# Classes: CapWords (PascalCase). Leading "_" allowed, no underscores inside.
CLASS_RE = re.compile(r"^_?[A-Z][a-zA-Z0-9]*$")

Row = Dict[str, str]

def find_python_dirs(root: Path):
    """Return directories that appear to contain Python code (have .py files)."""
    py_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if any(fn.endswith(".py") for fn in filenames):
            py_dirs.add(Path(dirpath))
    return sorted(py_dirs)

def classify_package_dir(d: Path, warn_underscore_packages: bool):
    rows = []
    name = d.name
    status = "OK"
    rule = "package_folder_name"
    msg = "Package/folder name is valid (lowercase)."
    if not PACKAGE_OK_RE.match(name):
        status = "ERROR"
        msg = "Folder contains Python code but name is not PEP 8 lowercase; avoid capitals/hyphens."
    elif "_" in name:
        if warn_underscore_packages:
            status = "WARN"
            msg = "Underscore in package name is discouraged by PEP 8."
        else:
            msg = "Underscore allowed (warning suppressed)."
    if name.startswith("_"):
        status = "WARN" if status == "OK" else status
        msg += " Leading underscore indicates non-public; ensure that's intentional."
    rows.append({
        "kind": "package", "path": str(d), "name": name,
        "rule": rule, "status": status, "message": msg
    })
    return rows

def classify_module_file(p: Path):
    rows = []
    name = p.name
    status = "OK"
    rule = "module_filename"
    msg = "Module filename is valid (lowercase, underscores allowed)."
    if name not in SPECIAL_MODULES and not MODULE_RE.match(name):
        status = "ERROR"
        msg = "Module filename should be all-lowercase; underscores allowed; no hyphens or capitals."
    if "-" in name:
        status = "ERROR"; msg = "Hyphens are not allowed in Python module names (use underscores)."
    if any(c.isupper() for c in name if c.isalpha()):
        status = "ERROR"; msg = "Uppercase letters are not allowed in module filenames."
    if name.startswith("_") and status == "OK":
        status = "WARN"; msg = "Leading underscore indicates non-public module; ensure that's intended."
    rows.append({
        "kind": "module", "path": str(p), "name": name,
        "rule": rule, "status": status, "message": msg
    })
    return rows

def classify_classes_in_file(p: Path):
    rows = []
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        rows.append({
            "kind": "file", "path": str(p), "name": p.name,
            "rule": "read_file", "status": "ERROR", "message": f"Failed to read file: {e}"
        })
        return rows
    try:
        tree = ast.parse(text, filename=str(p))
    except Exception as e:
        rows.append({
            "kind": "file", "path": str(p), "name": p.name,
            "rule": "parse_ast", "status": "ERROR", "message": f"Failed to parse Python file: {e}"
        })
        return rows

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls = node.name
            status = "OK"; msg = "Class name is valid CapWords (PascalCase)."
            if not CLASS_RE.match(cls) or "_" in cls:
                status = "ERROR"; msg = "Class name should use CapWords (PascalCase) with no underscores."
            elif cls.startswith("_"):
                status = "WARN"; msg = "Leading underscore indicates non-public class; ensure that's intended."
            rows.append({
                "kind": "class", "path": f"{p}:{node.lineno}", "name": cls,
                "rule": "class_name", "status": status, "message": msg
            })
    return rows

def sh_quote(p: Path) -> str:
    s = str(p)
    if any(ch in s for ch in [' ', '(', ')', '[', ']', '{', '}', '&']):
        return f"\"{s}\""
    return s

def suggest_module_name(name: str) -> str:
    # Strip extension and normalize to lower_snake
    if not name.endswith(".py"):
        return name
    base = name[:-3].replace("-", "_")
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base)   # camel/Pascal -> snake
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).lower().strip("_")
    if not base:
        base = "module"
    return base + ".py"

def suggest_package_name(name: str) -> str:
    base = name.replace("-", "_")
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base)
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base).lower().strip("_")
    if not base:
        base = "pkg"
    return base

def build_git_mv_plan(rows, root: Path) -> str:
    """Create a bash script with 'git mv' for ERROR module/package names (review before use)."""
    plan_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# Root: {root}"]
    for r in rows:
        if r["status"] != "ERROR":
            continue
        if r["kind"] == "module":
            src = Path(r["path"])
            tgt = suggest_module_name(src.name)
            if tgt and tgt != src.name:
                plan_lines.append(f"git mv {sh_quote(src)} {sh_quote(src.with_name(tgt))}")
        elif r["kind"] == "package":
            src = Path(r["path"])
            tgt = suggest_package_name(src.name)
            if tgt and tgt != src.name:
                plan_lines.append(f"git mv {sh_quote(src)} {sh_quote(src.with_name(tgt))}")
    return "\n".join(plan_lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="PEP 8 naming checker (modules, packages, classes).")
    parser.add_argument("path", nargs="?", default=".", help="Root path of the codebase (default: .)")
    parser.add_argument("--output", default="pep8_naming_report.csv", help="CSV output path")
    parser.add_argument("--no-warn-underscore-packages", action="store_true",
                        help="Do not warn about underscores in package/folder names")
    parser.add_argument("--git-mv-plan", action="store_true",
                        help="Also write a 'rename_plan.sh' with git mv commands for ERRORs")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    warn_underscore_packages = not args.no_warn_underscore_packages

    rows = []

    # Folders with Python code
    py_dirs = find_python_dirs(root)
    for d in py_dirs:
        rows.extend(classify_package_dir(d, warn_underscore_packages))

    # Modules + classes
    for d in py_dirs:
        for item in d.iterdir():
            if item.is_file() and item.suffix == ".py":
                rows.extend(classify_module_file(item))
                rows.extend(classify_classes_in_file(item))

    # CSV report
    out_path = Path(args.output).resolve()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "path", "name", "rule", "status", "message"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Optional rename plan
    if args.git_mv_plan:
        plan_text = build_git_mv_plan(rows, root)
        plan_path = root / "rename_plan.sh"
        with plan_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(plan_text)
        print(f"[+] Wrote git-mv plan: {plan_path}")

    total = len(rows)
    errors = sum(1 for r in rows if r["status"] == "ERROR")
    warns  = sum(1 for r in rows if r["status"] == "WARN")
    oks    = sum(1 for r in rows if r["status"] == "OK")
    print(f"[PEP8 Naming] Scanned: {root}")
    print(f"  Items: {total}  |  OK: {oks}  WARN: {warns}  ERROR: {errors}")
    print(f"[+] CSV report: {out_path}")
    if errors:
        print("  -> Tip: run with --git-mv-plan to generate a tentative rename script (review before use).")

if __name__ == "__main__":
    main()
