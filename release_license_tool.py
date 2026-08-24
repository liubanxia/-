from __future__ import annotations

import argparse
import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

from phoenix_knowledge.licensing import (
    LICENSE_VERSION,
    PRODUCT_ID,
    PRODUCT_NAME,
    build_activation_code,
)


def _load_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError("私钥必须是 Ed25519")
    return key


def generate_keys(output_dir: Path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = output_dir / "phoenix_license_private_key.pem"
    public_path = output_dir / "phoenix_license_public_key.pem"
    if private_path.exists() or public_path.exists():
        raise FileExistsError("密钥文件已存在，拒绝覆盖。请使用新的安全目录。")

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def issue_code(
    private_key_path: Path,
    machine_code: str,
    *,
    customer: str,
    edition: str,
    expires_at: str | None,
    features: list[str],
    license_id: str | None = None,
) -> str:
    private_key = _load_private_key(private_key_path)
    payload = {
        "version": LICENSE_VERSION,
        "product_id": PRODUCT_ID,
        "product_name": PRODUCT_NAME,
        "license_id": license_id or f"PHX-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(4).upper()}",
        "machine_code": machine_code.strip().upper(),
        "customer": customer.strip(),
        "edition": edition.strip() or "Professional",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at or None,
        "features": list(dict.fromkeys(features)),
    }
    return build_activation_code(private_key, payload)


def prepare_release(project_root: Path, public_key: Path, *, version: str, edition: str) -> None:
    root = Path(project_root).resolve()
    release_root = root / "16_产品发布"
    license_root = release_root / "授权"
    license_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(public_key, license_root / "license_public_key.pem")
    marker = {
        "product_id": PRODUCT_ID,
        "product_name": PRODUCT_NAME,
        "version": version,
        "edition": edition,
        "product_mode": True,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    (release_root / "PHOENIX_PRODUCT_RELEASE.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phoenix 正式版离线授权管理工具")
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser("keygen", help="首次正式发布时生成一对 Ed25519 密钥")
    keygen.add_argument("--out", required=True, help="私密密钥目录；不要放入Git仓库")

    issue = sub.add_parser("issue", help="根据客户机器码生成离线激活码")
    issue.add_argument("--private-key", required=True)
    issue.add_argument("--machine-code", required=True)
    issue.add_argument("--customer", default="")
    issue.add_argument("--edition", default="Professional")
    issue.add_argument("--expires", default="", help="YYYY-MM-DD；留空表示永久授权")
    issue.add_argument("--feature", action="append", default=[])
    issue.add_argument("--license-id", default="")
    issue.add_argument("--out", default="", help="可选：把激活码写入TXT文件")

    prepare = sub.add_parser("prepare-release", help="把公钥和正式版标记写入产品目录")
    prepare.add_argument("--project-root", required=True)
    prepare.add_argument("--public-key", required=True)
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--edition", default="Professional")
    return parser


def main() -> int:
    args = _parser().parse_args()

    if args.command == "keygen":
        private_path, public_path = generate_keys(Path(args.out))
        print(f"PRIVATE_KEY={private_path}")
        print(f"PUBLIC_KEY={public_path}")
        print("IMPORTANT=私钥只保存在你自己的授权电脑，不要提交Git，不要放进医院产品包。")
        return 0

    if args.command == "issue":
        code = issue_code(
            Path(args.private_key),
            args.machine_code,
            customer=args.customer,
            edition=args.edition,
            expires_at=args.expires.strip() or None,
            features=args.feature,
            license_id=args.license_id.strip() or None,
        )
        if args.out:
            Path(args.out).write_text(code + "\n", encoding="utf-8")
            print(f"ACTIVATION_FILE={Path(args.out).resolve()}")
        print(f"ACTIVATION_CODE={code}")
        return 0

    if args.command == "prepare-release":
        prepare_release(
            Path(args.project_root),
            Path(args.public_key),
            version=args.version,
            edition=args.edition,
        )
        print("PRODUCT_MODE=READY")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
