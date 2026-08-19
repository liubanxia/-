from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from phoenix_knowledge.licensing import (
    LICENSE_VERSION,
    PRODUCT_ID,
    LicenseManager,
    build_activation_code,
)


class LicensingTests(unittest.TestCase):
    def _configured_release(self, root: Path):
        release = root / "16_产品发布"
        license_root = release / "授权"
        license_root.mkdir(parents=True)
        (release / "PHOENIX_PRODUCT_RELEASE.json").write_text(
            json.dumps({"product_mode": True}),
            encoding="utf-8",
        )

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        (license_root / "license_public_key.pem").write_bytes(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return private_key, LicenseManager(root)

    def _payload(self, manager: LicenseManager, **overrides):
        payload = {
            "version": LICENSE_VERSION,
            "product_id": PRODUCT_ID,
            "license_id": "TEST-001",
            "machine_code": manager.machine_code,
            "customer": "Test Hospital",
            "edition": "Professional",
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
            "features": ["pdf_qa", "translation", "knowledge_organizer"],
        }
        payload.update(overrides)
        return payload

    def test_development_checkout_is_not_locked(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = LicenseManager(Path(temp))
            status = manager.status()
            self.assertTrue(status.valid)
            self.assertFalse(status.product_mode)
            self.assertEqual(status.edition, "Development")

    def test_release_requires_activation_then_accepts_signed_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key, manager = self._configured_release(root)
            before = manager.status()
            self.assertTrue(before.product_mode)
            self.assertFalse(before.valid)

            code = build_activation_code(private_key, self._payload(manager))
            activated = manager.activate(code)
            self.assertTrue(activated.valid)
            self.assertEqual(activated.license_id, "TEST-001")
            self.assertEqual(activated.customer, "Test Hospital")
            self.assertTrue(manager.activation_path.is_file())

            after = manager.status()
            self.assertTrue(after.valid)
            self.assertEqual(after.edition, "Professional")

    def test_code_is_bound_to_machine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key, manager = self._configured_release(root)
            code = build_activation_code(
                private_key,
                self._payload(manager, machine_code="PHX-0000-0000-0000-0000-0000-0000"),
            )
            with self.assertRaisesRegex(ValueError, "机器码"):
                manager.activate(code)

    def test_expired_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key, manager = self._configured_release(root)
            expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            code = build_activation_code(
                private_key,
                self._payload(manager, expires_at=expired),
            )
            with self.assertRaisesRegex(ValueError, "过期"):
                manager.activate(code)

    def test_tampered_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key, manager = self._configured_release(root)
            code = build_activation_code(private_key, self._payload(manager))
            prefix, payload, signature = code.split(".")
            tampered = f"{prefix}.{payload[:-1]}A.{signature}"
            with self.assertRaises((ValueError, RuntimeError)):
                manager.activate(tampered)


if __name__ == "__main__":
    unittest.main()
