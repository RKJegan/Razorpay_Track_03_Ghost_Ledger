"""
Ghost Ledger v3 — merchant registry (up to 10 merchants).

* At most :data:`config.MERCHANT_MAX_COUNT` merchants can exist. The limit is
  checked here for a clear error, and enforced again by a SQL trigger.
* API keys are generated once, shown to the operator once, and stored only as
  a SHA-256 hash. A lost key cannot be recovered, only rotated.
* Key comparison is constant-time.

Command line::

    python -m database.merchants create --id merchant_demo_001 --name "Demo Store"
    python -m database.merchants list
    python -m database.merchants rotate-key --id merchant_demo_001
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import secrets
from datetime import datetime

from sqlalchemy import func, select

from config import MERCHANT_MAX_COUNT
from database.engine import session_scope
from database.models import Merchant

logger = logging.getLogger(__name__)


class MerchantLimitError(RuntimeError):
    """Raised when creating a merchant would exceed :data:`MERCHANT_MAX_COUNT`."""


class DuplicateMerchantError(ValueError):
    """Raised when a merchant id is already registered."""


def _now() -> str:
    """Return the current local timestamp in the project's canonical format."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def hash_api_key(api_key: str) -> str:
    """
    Return the SHA-256 hex digest of an API key.

    Parameters
    ----------
    api_key : str
        Plain-text key as issued to the merchant.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """Return a new random API key (43 URL-safe characters, 256 bits)."""
    return secrets.token_urlsafe(32)


def count_merchants() -> int:
    """Return how many merchants are registered."""
    with session_scope() as session:
        return int(session.scalar(select(func.count()).select_from(Merchant)) or 0)


def create_merchant(merchant_id: str, name: str) -> str:
    """
    Register a merchant and return its new plain-text API key.

    The key is returned exactly once. Only its hash is stored.

    Parameters
    ----------
    merchant_id : str
        Stable identifier, e.g. ``merchant_demo_001``.
    name : str
        Display name.

    Returns
    -------
    str
        The plain-text API key. Show it to the operator, then discard it.

    Raises
    ------
    MerchantLimitError
        If the store already holds :data:`MERCHANT_MAX_COUNT` merchants.
    DuplicateMerchantError
        If ``merchant_id`` is already registered.
    """
    if not merchant_id or not merchant_id.strip():
        raise ValueError("merchant_id must be non-empty")
    if count_merchants() >= MERCHANT_MAX_COUNT:
        raise MerchantLimitError(
            f"merchant limit reached: at most {MERCHANT_MAX_COUNT} merchants"
        )
    api_key = generate_api_key()
    try:
        with session_scope() as session:
            session.add(
                Merchant(
                    id=merchant_id,
                    name=name,
                    api_key_hash=hash_api_key(api_key),
                    is_active=1,
                    created_at=_now(),
                )
            )
    except Exception as exc:
        # Distinguish the two expected failures from unexpected ones.
        if "UNIQUE" in str(exc).upper() or "PRIMARY KEY" in str(exc).upper():
            raise DuplicateMerchantError(f"merchant {merchant_id!r} already exists") from exc
        if "merchant limit reached" in str(exc):
            raise MerchantLimitError(str(exc)) from exc
        raise
    logger.info("merchant %s registered", merchant_id)
    return api_key


def rotate_api_key(merchant_id: str) -> str:
    """
    Issue a new API key for an existing merchant, invalidating the old one.

    Returns
    -------
    str
        The new plain-text key, shown once.

    Raises
    ------
    KeyError
        If the merchant does not exist.
    """
    api_key = generate_api_key()
    with session_scope() as session:
        merchant = session.get(Merchant, merchant_id)
        if merchant is None:
            raise KeyError(f"no merchant {merchant_id!r}")
        merchant.api_key_hash = hash_api_key(api_key)
    logger.info("api key rotated for merchant %s", merchant_id)
    return api_key


def verify_api_key(merchant_id: str, api_key: str | None) -> bool:
    """
    Return True when ``api_key`` belongs to an active merchant.

    Comparison is constant-time, and a missing or inactive merchant is treated
    the same as a wrong key, so the response reveals nothing about which
    merchant ids exist.
    """
    if not api_key:
        return False
    with session_scope() as session:
        merchant = session.get(Merchant, merchant_id)
        if merchant is None or not merchant.is_active:
            # Still do the comparison so timing does not reveal existence.
            hmac.compare_digest(hash_api_key(api_key), hash_api_key(""))
            return False
        return hmac.compare_digest(merchant.api_key_hash, hash_api_key(api_key))


def list_merchants() -> list[dict[str, object]]:
    """Return every merchant (never the key hash)."""
    with session_scope() as session:
        rows = session.scalars(select(Merchant).order_by(Merchant.id)).all()
        return [
            {"id": m.id, "name": m.name, "is_active": bool(m.is_active), "created_at": m.created_at}
            for m in rows
        ]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for merchant administration."""
    from database.db_client import init_db

    parser = argparse.ArgumentParser(description="Ghost Ledger merchant registry")
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create", help="register a merchant (prints its API key once)")
    create.add_argument("--id", required=True)
    create.add_argument("--name", required=True)
    rotate = sub.add_parser("rotate-key", help="issue a new API key (old key stops working)")
    rotate.add_argument("--id", required=True)
    sub.add_parser("list", help="list merchants (no keys)")
    args = parser.parse_args(argv)

    init_db()  # v2 schema + v3 migrations, so the audit trail exists too
    if args.cmd == "create":
        key = create_merchant(args.id, args.name)
        print(f"merchant {args.id} registered.")
        print("API key (shown once, store it now):")
        print(key)
    elif args.cmd == "rotate-key":
        print("new API key (shown once):")
        print(rotate_api_key(args.id))
    else:
        for m in list_merchants():
            print(f"  {m['id']:<22} {m['name']:<28} active={m['is_active']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
