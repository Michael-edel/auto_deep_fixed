#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

from utils.logger import setup_logging
from utils.config import load_settings
from core.ai_client import AIClient
from storage.database import Database
from core.document_processor import DocumentProcessor


def _safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.strip().strip(".")
    return name or "document"


def _iter_input_files(p: Path):
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    if p.is_file():
        yield p
        return
    if p.is_dir():
        for fp in sorted(p.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in exts:
                yield fp
        return
    raise FileNotFoundError(f"Не найдено: {p}")


def main():
    parser = argparse.ArgumentParser(description="Распознавание документов (JPG/PNG/PDF) -> JSON")
    parser.add_argument("file", help="Путь к файлу (jpg/png/pdf) ИЛИ папке с файлами")
    parser.add_argument("--out", default="out/result.json", help="Куда сохранить JSON (для папки: папка out/)")
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования")
    args = parser.parse_args()

    setup_logging(args.log_level)

    settings = load_settings()
    ai = AIClient(api_key=settings.openai_api_key, model=settings.openai_model)
    db = Database(settings.db_path)
    processor = DocumentProcessor(ai=ai, db=db)

    in_path = Path(args.file)
    files = list(_iter_input_files(in_path))
    if not files:
        raise ValueError(f"В папке нет поддерживаемых файлов: {in_path}")

    is_batch = in_path.is_dir() or len(files) > 1

    if not is_batch:
        result = processor.process_file(files[0])
        out_path = processor.save_json(result, Path(args.out))
        print(f"✅ Готово: {out_path}")
        return

    out_arg = Path(args.out)
    out_dir = out_arg if out_arg.suffix.lower() != ".json" else out_arg.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for fp in files:
        try:
            result = processor.process_file(fp)
            error = ""
        except Exception as e:
            result = None
            error = f"{type(e).__name__}: {e}"

        per_name = _safe_name(fp.stem) + ".json"
        per_path = out_dir / per_name

        payload = {"source_file": str(fp), "error": error, "result": result}

        with per_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        documents.append(payload)

    summary_path = out_dir / "result_all.json"
    summary = {"input": str(in_path), "count": len(files), "documents": documents}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово: {summary_path}")
    print(f"✅ Сохранены отдельные JSON для {len(files)} файлов в: {out_dir}")


if __name__ == "__main__":
    main()
