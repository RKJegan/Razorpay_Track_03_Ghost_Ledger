"""
Tests for the merchant registry: the 10-merchant cap, hashed keys, and
constant-time key verification (database/merchants.py).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from config import MERCHANT_MAX_COUNT
from database import merchants
from database.engine import session_scope
from database.models import Merchant


def _delete_merchant(merchant_id: str) -> None:
    """Remove a test merchant. Deleting merchants is allowed; only events are immutable."""
    with session_scope() as session:
        session.execute(text("DELETE FROM merchants WHERE id = :i"), {"i": merchant_id})


def test_cap_is_ten() -> None:
    assert MERCHANT_MAX_COUNT == 10


def test_cap_is_enforced_at_eleven() -> None:
    """The 11th merchant is refused, by the service AND by the database trigger."""
    created: list[str] = []
    try:
        existing = merchants.count_merchants()
        for i in range(MERCHANT_MAX_COUNT - existing):
            mid = f"m_cap_{i:02d}"
            merchants.create_merchant(mid, f"Cap test {i}")
            created.append(mid)
        assert merchants.count_merchants() == MERCHANT_MAX_COUNT

        with pytest.raises(merchants.MerchantLimitError):
            merchants.create_merchant("m_cap_overflow", "Too many")

        # Bypass the service: the trigger must still refuse the row.
        with pytest.raises(Exception) as excinfo:
            with session_scope() as session:
                session.add(
                    Merchant(
                        id="m_cap_direct",
                        name="direct insert",
                        api_key_hash="x",
                        is_active=1,
                        created_at="2026-10-08 00:00:00",
                    )
                )
        assert "merchant limit" in str(excinfo.value)
        assert merchants.count_merchants() == MERCHANT_MAX_COUNT
    finally:
        for mid in created:
            _delete_merchant(mid)


def test_api_key_is_shown_once_and_stored_hashed() -> None:
    mid = "m_hash_check"
    key = merchants.create_merchant(mid, "Hash check")
    try:
        with session_scope() as session:
            row = session.get(Merchant, mid)
            assert row is not None
            assert row.api_key_hash != key
            assert row.api_key_hash == merchants.hash_api_key(key)
        assert merchants.verify_api_key(mid, key)
        assert not merchants.verify_api_key(mid, key + "x")
        assert not merchants.verify_api_key(mid, "")
        assert not merchants.verify_api_key(mid, None)
        assert not merchants.verify_api_key("m_does_not_exist", key)
    finally:
        _delete_merchant(mid)


def test_rotating_key_invalidates_the_old_one() -> None:
    mid = "m_rotate_check"
    old = merchants.create_merchant(mid, "Rotate check")
    try:
        new = merchants.rotate_api_key(mid)
        assert new != old
        assert not merchants.verify_api_key(mid, old)
        assert merchants.verify_api_key(mid, new)
    finally:
        _delete_merchant(mid)


def test_duplicate_merchant_is_refused() -> None:
    mid = "m_duplicate_check"
    merchants.create_merchant(mid, "First")
    try:
        with pytest.raises(merchants.DuplicateMerchantError):
            merchants.create_merchant(mid, "Second")
    finally:
        _delete_merchant(mid)


def test_list_never_exposes_key_material() -> None:
    mid = "m_list_check"
    merchants.create_merchant(mid, "List check")
    try:
        listed = [m for m in merchants.list_merchants() if m["id"] == mid]
        assert listed
        assert set(listed[0]) == {"id", "name", "is_active", "created_at"}
    finally:
        _delete_merchant(mid)
