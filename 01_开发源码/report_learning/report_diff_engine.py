from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import hashlib
import re


SECTION_ALIASES = {
    "检查所见": {
        "检查所见",
        "影像学所见",
        "影像所见",
        "所见",
        "影像表现",
    },
    "诊断意见": {
        "诊断意见",
        "影像诊断",
        "诊断",
        "印象",
        "结论",
    },
}


@dataclass
class ReportChange:
    section: str
    change_type: str
    ai_text: str
    final_text: str

    def to_dict(self):
        return asdict(self)


class ReportDiffEngine:

    def normalize_text(self, text):
        text = str(text or "")

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = text.replace("\u3000", " ")

        lines = []

        for line in text.splitlines():
            line = re.sub(
                r"[ \t]+",
                " ",
                line.strip(),
            )

            if line:
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _hash(text):
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _canonical_heading(value):
        key = re.sub(
            r"[\s：:【】\[\]]+",
            "",
            value,
        )

        for canonical, aliases in SECTION_ALIASES.items():
            cleaned = {
                re.sub(
                    r"[\s：:【】\[\]]+",
                    "",
                    x,
                )
                for x in aliases
            }

            if key in cleaned:
                return canonical

        return None

    def split_sections(self, text):
        text = self.normalize_text(text)

        sections = {
            "其他": [],
            "检查所见": [],
            "诊断意见": [],
        }

        current = "其他"

        for line in text.splitlines():
            line = line.strip()

            match = re.match(
                r"^[【\[]?([^：:\]】]{1,12})[\]】]?\s*[：:]?\s*(.*)$",
                line,
            )

            heading = None
            remainder = ""

            if match:
                heading = self._canonical_heading(
                    match.group(1)
                )
                remainder = match.group(2).strip()

            if heading:
                current = heading

                if remainder:
                    sections[current].append(
                        remainder
                    )
                continue

            sections[current].append(line)

        return {
            key: "\n".join(value).strip()
            for key, value in sections.items()
            if value
        }

    def split_units(self, text):
        text = self.normalize_text(text)

        if not text:
            return []

        units = re.split(
            r"(?<=[。；;！？!?])\s*|\n+",
            text,
        )

        return [
            x.strip()
            for x in units
            if x.strip()
        ]

    def _diff_section(
        self,
        section,
        ai_text,
        final_text,
    ):
        ai_units = self.split_units(
            ai_text
        )

        final_units = self.split_units(
            final_text
        )

        matcher = SequenceMatcher(
            a=ai_units,
            b=final_units,
            autojunk=False,
        )

        changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():

            if tag == "equal":
                continue

            ai_chunk = "\n".join(
                ai_units[i1:i2]
            )

            final_chunk = "\n".join(
                final_units[j1:j2]
            )

            if tag == "insert":
                change_type = "add"

            elif tag == "delete":
                change_type = "delete"

            else:
                change_type = "replace"

            changes.append(
                ReportChange(
                    section=section,
                    change_type=change_type,
                    ai_text=ai_chunk,
                    final_text=final_chunk,
                )
            )

        return changes

    def compare(
        self,
        ai_draft,
        final_report,
    ):
        ai_norm = self.normalize_text(
            ai_draft
        )

        final_norm = self.normalize_text(
            final_report
        )

        ai_sections = self.split_sections(
            ai_norm
        )

        final_sections = self.split_sections(
            final_norm
        )

        changes = []

        for section in [
            "检查所见",
            "诊断意见",
            "其他",
        ]:
            changes.extend(
                self._diff_section(
                    section,
                    ai_sections.get(
                        section,
                        ""
                    ),
                    final_sections.get(
                        section,
                        ""
                    ),
                )
            )

        similarity = SequenceMatcher(
            a=ai_norm,
            b=final_norm,
            autojunk=False,
        ).ratio()

        return {
            "schema": "phoenix.report_diff.v1",

            "exact_match":
                ai_norm == final_norm,

            "similarity":
                round(
                    float(similarity),
                    6,
                ),

            "ai_hash":
                self._hash(
                    ai_norm
                ),

            "final_hash":
                self._hash(
                    final_norm
                ),

            "summary": {
                "change_count":
                    len(changes),

                "add_count":
                    sum(
                        x.change_type == "add"
                        for x in changes
                    ),

                "delete_count":
                    sum(
                        x.change_type == "delete"
                        for x in changes
                    ),

                "replace_count":
                    sum(
                        x.change_type == "replace"
                        for x in changes
                    ),
            },

            "changes": [
                x.to_dict()
                for x in changes
            ],
        }
