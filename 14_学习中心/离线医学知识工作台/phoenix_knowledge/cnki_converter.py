from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


CNKI_EXTENSIONS = {".caj", ".nh", ".hn", ".kdh", ".teb", ".c8"}


@dataclass(frozen=True)
class CNKIConverterStatus:
    available: bool
    backend: str
    executable: str
    warning: str = ""

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "backend": self.backend,
            "executable": self.executable,
            "warning": self.warning,
            "extensions": sorted(CNKI_EXTENSIONS),
        }


class CNKIConversionError(RuntimeError):
    pass


class CNKIConverter:
    """Offline CAJ/KDH/NH/TEB compatibility gateway.

    Product builds can bundle a caj2pdf-compatible CLI. Phoenix never uploads
    the source file and caches the converted PDF by source SHA-256.
    """

    def __init__(self, paths):
        self.paths = paths
        self.cache_root = Path(paths.runtime_root) / "cnki_pdf_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while True:
                block = handle.read(block_size)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _candidates(self) -> list[Path]:
        result: list[Path] = []
        env = os.environ.get("PHOENIX_CNKI_CONVERTER", "").strip()
        if env:
            result.append(Path(env))
        project = Path(self.paths.project_root)
        runtime = Path(self.paths.runtime_root)
        result.extend(
            [
                project / "02_开发环境" / "caj2pdf" / "caj2pdf.exe",
                project / "02_开发环境" / "caj2pdf" / "caj2pdf",
                project / "01_开发源码" / "third_party" / "caj2pdf" / "caj2pdf.exe",
                runtime / "tools" / "caj2pdf" / "caj2pdf.exe",
                runtime / "tools" / "caj2pdf" / "caj2pdf",
            ]
        )
        for name in ("caj2pdf.exe", "caj2pdf", "caj2pdf-rs.exe", "caj2pdf-rs"):
            found = shutil.which(name)
            if found:
                result.append(Path(found))
        unique: list[Path] = []
        seen = set()
        for item in result:
            key = str(item).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def status(self) -> CNKIConverterStatus:
        for candidate in self._candidates():
            if candidate.is_file():
                return CNKIConverterStatus(
                    available=True,
                    backend="caj2pdf-cli",
                    executable=str(candidate.resolve()),
                )
        return CNKIConverterStatus(
            available=False,
            backend="none",
            executable="",
            warning=(
                "未检测到本地CAJ转换器。正式安装包应随产品附带"
                "caj2pdf兼容CLI；原始论文不会上传网络。"
            ),
        )

    @staticmethod
    def _pdf_valid(path: Path) -> bool:
        path = Path(path)
        if not path.is_file() or path.stat().st_size < 64:
            return False
        try:
            head = path.read_bytes()[:16]
        except OSError:
            return False
        if b"%PDF-" not in head:
            return False
        try:
            import pymupdf
            doc = pymupdf.open(path)
            ok = len(doc) > 0
            doc.close()
            return ok
        except Exception:
            try:
                import fitz
                doc = fitz.open(path)
                ok = len(doc) > 0
                doc.close()
                return ok
            except Exception:
                return True

    def _embedded_pdf(self, source: Path, target: Path) -> bool:
        """Best-effort extraction for containers that directly embed a PDF."""
        try:
            payload = Path(source).read_bytes()
        except OSError:
            return False
        start = payload.find(b"%PDF-")
        end = payload.rfind(b"%%EOF")
        if start < 0 or end <= start:
            return False
        temp = target.with_suffix(".embedded.tmp.pdf")
        temp.write_bytes(payload[start : end + len(b"%%EOF")])
        if self._pdf_valid(temp):
            temp.replace(target)
            return True
        temp.unlink(missing_ok=True)
        return False

    @staticmethod
    def _timeout() -> int:
        try:
            value = int(os.environ.get("PHOENIX_CNKI_CONVERT_TIMEOUT", "300"))
        except (TypeError, ValueError):
            value = 300
        return max(30, min(value, 1800))

    def convert(self, source: Path) -> Path:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix not in CNKI_EXTENSIONS:
            raise ValueError(f"不是支持的知网专有格式: {source.suffix}")

        digest = self._sha256(source)
        target = self.cache_root / f"{source.stem[:72]}_{digest[:16]}.pdf"
        if self._pdf_valid(target):
            return target

        if self._embedded_pdf(source, target):
            return target

        status = self.status()
        if not status.available:
            raise CNKIConversionError(
                f"{source.suffix.upper()} 已进入 Phoenix 格式体系，但本机缺少离线转换后端。"
                "请使用随正式产品打包的 caj2pdf 兼容组件。"
            )

        executable = Path(status.executable)
        temp = target.with_suffix(".converting.pdf")
        temp.unlink(missing_ok=True)
        cmd = [
            str(executable),
            "convert",
            str(source),
            "-o",
            str(temp),
        ]
        timeout = self._timeout()
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.cache_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            temp.unlink(missing_ok=True)
            raise CNKIConversionError(
                f"{source.name} 转换超过 {timeout} 秒，已安全终止。"
            ) from exc
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise CNKIConversionError(
                f"无法启动本地CAJ转换器: {type(exc).__name__}: {exc}"
            ) from exc

        if completed.returncode != 0 or not self._pdf_valid(temp):
            temp.unlink(missing_ok=True)
            detail = (completed.stdout or "").strip()[-1200:]
            format_hint = ""
            if suffix == ".teb":
                format_hint = (
                    "；TEB兼容性取决于产品内置转换后端版本，"
                    "Phoenix不会把转换失败伪装成导入成功"
                )
            raise CNKIConversionError(
                f"{source.name} 离线转换失败(exit={completed.returncode})"
                f"{format_hint}。{detail}"
            )

        temp.replace(target)
        return target
