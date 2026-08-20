from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import WorkbenchPaths


class LegacyPPTConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LegacyPPTStatus:
    available: bool
    backend: str
    executable: str = ""
    bundled: bool = False
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class LegacyPPTConverter:
    """Convert binary PowerPoint ``.ppt`` files to cached ``.pptx`` offline.

    Priority:
    1. Phoenix-bundled / explicitly configured LibreOffice.
    2. System LibreOffice.
    3. Microsoft PowerPoint COM automation on Windows, but only when the
       PowerPoint COM class is actually registered on the machine.

    The conversion result is cached by source SHA-256, so the expensive
    compatibility conversion happens only once for an unchanged presentation.
    """

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.cache_root = Path(paths.runtime_root) / "legacy_office_cache" / "ppt"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _libreoffice_candidates(self) -> list[tuple[Path, bool]]:
        candidates: list[tuple[Path, bool]] = []

        configured = os.environ.get("PHOENIX_LIBREOFFICE", "").strip()
        if configured:
            candidates.append((Path(configured), False))

        root = Path(self.paths.project_root)
        bundled = (
            root / "02_开发环境" / "LibreOffice" / "program" / "soffice.exe",
            root / "02_开发环境" / "LibreOffice" / "program" / "soffice.com",
            root / "14_学习中心" / "离线医学知识工作台" / "vendor" / "LibreOffice" / "program" / "soffice.exe",
            root / "14_学习中心" / "离线医学知识工作台" / "vendor" / "LibreOffice" / "program" / "soffice.com",
        )
        candidates.extend((path, True) for path in bundled)

        for name in ("soffice", "soffice.com", "libreoffice"):
            found = shutil.which(name)
            if found:
                candidates.append((Path(found), False))

        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(env_name, "").strip()
            if base:
                candidates.append(
                    (Path(base) / "LibreOffice" / "program" / "soffice.exe", False)
                )
                candidates.append(
                    (Path(base) / "LibreOffice" / "program" / "soffice.com", False)
                )

        unique: list[tuple[Path, bool]] = []
        seen: set[str] = set()
        for path, is_bundled in candidates:
            try:
                key = str(path.expanduser().resolve()).lower()
            except Exception:
                key = str(path).lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append((path.expanduser(), is_bundled))
        return unique

    def _find_libreoffice(self) -> tuple[Path, bool] | None:
        for path, bundled in self._libreoffice_candidates():
            try:
                if path.is_file():
                    return path.resolve(), bundled
            except OSError:
                continue
        return None

    @staticmethod
    def _find_powershell() -> Path | None:
        if os.name != "nt":
            return None
        for name in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
            found = shutil.which(name)
            if found:
                return Path(found)
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        candidate = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        return candidate if candidate.is_file() else None

    @staticmethod
    def _powerpoint_registered() -> bool:
        """Return True only when Microsoft PowerPoint COM is really installed.

        PowerShell is present on most Windows systems and is not evidence that
        Office/PowerPoint exists. Checking the COM ProgID prevents the GUI from
        advertising legacy-PPT support that will only fail at import time.
        """
        if os.name != "nt":
            return False
        try:
            import winreg
        except ImportError:
            return False

        candidates = (
            r"PowerPoint.Application\CLSID",
            r"WOW6432Node\Classes\PowerPoint.Application\CLSID",
        )
        roots = (winreg.HKEY_CLASSES_ROOT, winreg.HKEY_LOCAL_MACHINE)
        for root in roots:
            for key_name in candidates:
                try:
                    with winreg.OpenKey(root, key_name):
                        return True
                except OSError:
                    continue
        return False

    @classmethod
    def _find_powerpoint_powershell(cls) -> Path | None:
        if not cls._powerpoint_registered():
            return None
        return cls._find_powershell()

    def status(self) -> LegacyPPTStatus:
        libreoffice = self._find_libreoffice()
        if libreoffice is not None:
            executable, bundled = libreoffice
            return LegacyPPTStatus(
                available=True,
                backend="libreoffice",
                executable=str(executable),
                bundled=bundled,
                message="老式PPT可自动兼容转换",
            )

        powershell = self._find_powerpoint_powershell()
        if powershell is not None:
            return LegacyPPTStatus(
                available=True,
                backend="powerpoint_com",
                executable=str(powershell),
                bundled=False,
                message="已检测到本机 Microsoft PowerPoint，可自动转换老式PPT",
            )

        return LegacyPPTStatus(
            available=False,
            backend="unavailable",
            message=(
                "当前电脑没有可用的老式PPT转换组件。正式版可随程序附带LibreOffice兼容组件，"
                "或使用已安装并注册COM组件的Microsoft PowerPoint。"
            ),
        )

    def _cache_target(self, source: Path) -> Path:
        digest = self._file_digest(source)
        folder = self.cache_root / digest[:20]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{source.stem}.pptx"

    @staticmethod
    def _run_flags() -> int:
        return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0

    def _convert_with_libreoffice(self, executable: Path, source: Path, target: Path) -> None:
        workdir = target.parent
        profile_dir = workdir / "lo_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_uri = profile_dir.resolve().as_uri()

        command = [
            str(executable),
            "--headless",
            "--nologo",
            "--nodefault",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation={profile_uri}",
            "--convert-to",
            "pptx:Impress MS PowerPoint 2007 XML",
            "--outdir",
            str(workdir),
            str(source),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=180,
            creationflags=self._run_flags(),
            check=False,
        )

        generated = workdir / f"{source.stem}.pptx"
        if not generated.is_file():
            pptx_files = sorted(workdir.glob("*.pptx"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pptx_files:
                generated = pptx_files[0]

        if completed.returncode != 0 or not generated.is_file() or generated.stat().st_size <= 0:
            detail = (completed.stderr or completed.stdout or "LibreOffice未生成PPTX").strip()
            raise LegacyPPTConversionError(f"LibreOffice兼容转换失败：{detail[-600:]}")

        if generated.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(generated), str(target))

    def _convert_with_powerpoint(self, powershell: Path, source: Path, target: Path) -> None:
        script = target.parent / "convert_legacy_ppt.ps1"
        script.write_text(
            r'''param([string]$Source,[string]$Target)
$ErrorActionPreference = 'Stop'
$ppt = $null
$presentation = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $presentation = $ppt.Presentations.Open($Source, $true, $true, $false)
    $presentation.SaveAs($Target, 24)
}
finally {
    if ($presentation -ne $null) { try { $presentation.Close() } catch {} }
    if ($ppt -ne $null) { try { $ppt.Quit() } catch {} }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
''',
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                str(source),
                str(target),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=180,
            creationflags=self._run_flags(),
            check=False,
        )
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
            detail = (completed.stderr or completed.stdout or "PowerPoint未生成PPTX").strip()
            raise LegacyPPTConversionError(f"Microsoft PowerPoint兼容转换失败：{detail[-600:]}")

    def convert(self, source: Path) -> Path:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".ppt":
            raise ValueError(f"不是老式PowerPoint文件: {source}")

        target = self._cache_target(source)
        if target.is_file() and target.stat().st_size > 0:
            return target

        errors: list[str] = []
        libreoffice = self._find_libreoffice()
        if libreoffice is not None:
            try:
                self._convert_with_libreoffice(libreoffice[0], source, target)
                return target
            except Exception as exc:
                errors.append(str(exc))

        powershell = self._find_powerpoint_powershell()
        if powershell is not None:
            try:
                self._convert_with_powerpoint(powershell, source, target)
                return target
            except Exception as exc:
                errors.append(str(exc))

        detail = "；".join(errors[-2:]) if errors else "没有检测到可用转换组件"
        raise LegacyPPTConversionError(
            "无法读取老式 .ppt。Phoenix 已自动尝试兼容转换，但当前环境没有成功完成转换。"
            "正式产品包应附带 LibreOffice 兼容组件；如果电脑已安装 Microsoft PowerPoint，"
            "Phoenix 也会自动调用。无需用户手工另存。"
            f" 详细信息：{detail}"
        )
