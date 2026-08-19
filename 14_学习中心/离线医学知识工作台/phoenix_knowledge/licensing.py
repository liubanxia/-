from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRODUCT_ID = "phoenix.medical.knowledge.workbench"
PRODUCT_NAME = "Phoenix 医学知识工作台"
LICENSE_VERSION = 1
_RELEASE_MARKER = "PHOENIX_PRODUCT_RELEASE.json"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * ((4 - len(text) % 4) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def product_mode_enabled(project_root: Path) -> bool:
    """Return True only for an explicit product/release build.

    Development checkouts stay unlocked. A final packaged release becomes locked
    either by setting PHOENIX_PRODUCT_MODE=1 or by shipping the release marker
    created by release_license_tool.py.
    """

    if _truthy(os.environ.get("PHOENIX_PRODUCT_MODE")):
        return True
    return (Path(project_root) / "16_产品发布" / _RELEASE_MARKER).is_file()


def _windows_machine_guid() -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        access = winreg.KEY_READ
        wow64 = getattr(winreg, "KEY_WOW64_64KEY", 0)
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            access | wow64,
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "MachineGuid")
        return str(value).strip()
    except Exception:
        return ""


def machine_code(product_id: str = PRODUCT_ID) -> str:
    """Build a stable, non-reversible machine identifier for offline licensing."""

    machine_guid = _windows_machine_guid()
    fallback = "|".join(
        [
            platform.node(),
            platform.machine(),
            platform.processor(),
            str(uuid.getnode()),
            os.environ.get("PROCESSOR_IDENTIFIER", ""),
        ]
    )
    raw = f"{product_id}|{machine_guid or fallback}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(raw).hexdigest().upper()[:24]
    groups = "-".join(digest[index : index + 4] for index in range(0, len(digest), 4))
    return f"PHX-{groups}"


def _parse_expiry(value: Any) -> datetime | None:
    if value in (None, "", 0):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text += "T23:59:59+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class LicenseStatus:
    valid: bool
    product_mode: bool
    configured: bool
    machine_code: str
    message: str
    license_id: str = ""
    customer: str = ""
    edition: str = ""
    expires_at: str = ""
    features: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "product_mode": self.product_mode,
            "configured": self.configured,
            "machine_code": self.machine_code,
            "message": self.message,
            "license_id": self.license_id,
            "customer": self.customer,
            "edition": self.edition,
            "expires_at": self.expires_at,
            "features": list(self.features),
        }


class LicenseManager:
    def __init__(self, project_root: Path, *, product_id: str = PRODUCT_ID):
        self.project_root = Path(project_root).resolve()
        self.product_id = product_id
        self.release_root = self.project_root / "16_产品发布"
        self.license_root = self.release_root / "授权"
        self.activation_path = self.license_root / "activation.lic"
        self.public_key_path = Path(
            os.environ.get("PHOENIX_LICENSE_PUBLIC_KEY_FILE")
            or self.license_root / "license_public_key.pem"
        )
        self.release_marker = self.release_root / _RELEASE_MARKER

    @property
    def machine_code(self) -> str:
        return machine_code(self.product_id)

    @property
    def product_mode(self) -> bool:
        return product_mode_enabled(self.project_root)

    @property
    def configured(self) -> bool:
        return self.public_key_path.is_file()

    def _load_public_key(self):
        if not self.public_key_path.is_file():
            raise RuntimeError(
                "正式版授权公钥尚未配置。请先使用 release_license_tool.py prepare-release。"
            )
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except Exception as exc:
            raise RuntimeError(
                "缺少授权验证依赖 cryptography；正式产品环境必须安装 requirements-base.txt。"
            ) from exc

        key = serialization.load_pem_public_key(self.public_key_path.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            raise RuntimeError("授权公钥格式错误：必须是 Ed25519 公钥")
        return key

    def verify_code(self, code: str) -> dict[str, Any]:
        code = "".join((code or "").split())
        parts = code.split(".")
        if len(parts) != 3 or parts[0] != "PHX1":
            raise ValueError("激活码格式无效")

        try:
            payload_bytes = _b64url_decode(parts[1])
            signature = _b64url_decode(parts[2])
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            raise ValueError("激活码内容损坏") from exc

        public_key = self._load_public_key()
        try:
            public_key.verify(signature, payload_bytes)
        except Exception as exc:
            raise ValueError("激活码签名无效") from exc

        if int(payload.get("version", 0) or 0) != LICENSE_VERSION:
            raise ValueError("激活码版本不兼容")
        if str(payload.get("product_id", "")) != self.product_id:
            raise ValueError("激活码不属于当前产品")

        licensed_machine = str(payload.get("machine_code", "")).strip().upper()
        if licensed_machine not in {self.machine_code.upper(), "*"}:
            raise ValueError("激活码与当前机器码不匹配")

        license_id = str(payload.get("license_id", "")).strip()
        if not license_id:
            raise ValueError("激活码缺少授权编号")

        expiry = _parse_expiry(payload.get("expires_at"))
        if expiry is not None and datetime.now(timezone.utc) > expiry:
            raise ValueError("授权已过期")

        return payload

    def activate(self, code: str) -> LicenseStatus:
        payload = self.verify_code(code)
        self.license_root.mkdir(parents=True, exist_ok=True)
        record = {
            "activation_code": "".join((code or "").split()),
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "machine_code": self.machine_code,
            "payload": payload,
        }
        temp = self.activation_path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(self.activation_path)
        return self.status()

    def deactivate(self) -> None:
        self.activation_path.unlink(missing_ok=True)

    def _status_from_payload(self, payload: dict[str, Any], message: str) -> LicenseStatus:
        features_raw = payload.get("features", [])
        features = tuple(str(item) for item in features_raw) if isinstance(features_raw, list) else ()
        return LicenseStatus(
            valid=True,
            product_mode=True,
            configured=True,
            machine_code=self.machine_code,
            message=message,
            license_id=str(payload.get("license_id", "")),
            customer=str(payload.get("customer", "")),
            edition=str(payload.get("edition", "Professional")),
            expires_at=str(payload.get("expires_at") or "永久"),
            features=features,
        )

    def status(self) -> LicenseStatus:
        if not self.product_mode:
            return LicenseStatus(
                valid=True,
                product_mode=False,
                configured=self.configured,
                machine_code=self.machine_code,
                message="开发模式：当前不启用产品激活锁。",
                edition="Development",
            )

        if not self.configured:
            return LicenseStatus(
                valid=False,
                product_mode=True,
                configured=False,
                machine_code=self.machine_code,
                message="正式版授权公钥尚未配置。",
            )

        if not self.activation_path.is_file():
            return LicenseStatus(
                valid=False,
                product_mode=True,
                configured=True,
                machine_code=self.machine_code,
                message="产品尚未激活。",
            )

        try:
            record = json.loads(self.activation_path.read_text(encoding="utf-8"))
            code = str(record.get("activation_code", ""))
            payload = self.verify_code(code)
            return self._status_from_payload(payload, "授权有效。")
        except Exception as exc:
            return LicenseStatus(
                valid=False,
                product_mode=True,
                configured=True,
                machine_code=self.machine_code,
                message=f"授权无效：{type(exc).__name__}: {exc}",
            )

    def require_active(self) -> LicenseStatus:
        status = self.status()
        if not status.valid:
            raise RuntimeError(
                f"{status.message} 当前机器码：{status.machine_code}"
            )
        return status


def build_activation_code(private_key, payload: dict[str, Any]) -> str:
    """Sign one canonical payload. Used by the private release issuer tool."""

    payload_bytes = _canonical_json(payload)
    signature = private_key.sign(payload_bytes)
    return "PHX1." + _b64url_encode(payload_bytes) + "." + _b64url_encode(signature)
