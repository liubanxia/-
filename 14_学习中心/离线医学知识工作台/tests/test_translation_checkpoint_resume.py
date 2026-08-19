from __future__ import annotations

import json
from pathlib import Path

from phoenix_knowledge.translator import PDFTranslator


def test_existing_checkpoint_start_page_is_authoritative(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "source_sha256": "abc",
                "target_language": "中文",
                "start_page": 37,
                "last_completed_page": 42,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = PDFTranslator._read_json(checkpoint)
    requested_start_page = 1

    resolved = PDFTranslator._resolve_resume_start_page(
        state,
        requested_start_page,
        force_restart=False,
    )

    assert resolved == 37


def test_force_restart_keeps_requested_start_page():
    state = {"start_page": 37}
    assert PDFTranslator._resolve_resume_start_page(
        state,
        12,
        force_restart=True,
    ) == 12
