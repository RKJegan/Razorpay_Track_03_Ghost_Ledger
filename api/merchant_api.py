"""
Ghost Ledger v3 — merchant-scoped API and web view (A6).

    GET /api/merchant/me                      who the key belongs to
    GET /api/merchant/summary                 this merchant's live state
    GET /api/merchant/recoveries              this merchant's recoveries (filters, paging)
    GET /api/merchant/recoveries/{id}         one recovery plus its timeline
    GET /merchant                             the HTML/JS dashboard (static files)

Every route needs ``Authorization: Bearer <merchant API key>``. The key identifies
the merchant. Each query filters by that merchant id, in the query itself, not in
the page. A recovery belonging to another merchant returns 404, the same answer
as an id that does not exist, so ids cannot be probed.

Operator-only details (who approved, rejection reasons, reference ids) are not
returned. The timeline shows stage names and a whitelist of detail fields.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse

from api.operator_api import FAILURE_STAGES
from database import audit_trail, approvals, merchants, recovery_store

router = APIRouter(prefix="/api/merchant", tags=["merchant"])
page_router = APIRouter(tags=["merchant"])

WEB_DIR = Path(__file__).resolve().parents[1] / "web" / "merchant"

# Detail fields a merchant may see. Everything else stays internal.
SAFE_DETAIL_KEYS: frozenset[str] = frozenset({
    "payment_id", "amount_paise", "attempt", "short_url", "expire_by", "link_id",
})

# Approval fields a merchant may see.
SAFE_APPROVAL_KEYS: tuple[str, ...] = (
    "approval_id", "recovery_id", "amount_inr", "cause", "status", "created_at",
)


def require_merchant(
    response: Response,
    authorization: str | None = Header(default=None),
) -> str:
    """
    Resolve the merchant from the bearer key. Returns the merchant id.

    Raises
    ------
    HTTPException
        401 if the header is missing, malformed, or the key matches no active merchant.
    """
    response.headers["Cache-Control"] = "no-store"
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    merchant_id = merchants.merchant_for_api_key(token)
    if merchant_id is None:
        # Log that a rejection happened. Never log the key itself.
        audit_trail.log(
            component="merchants",
            action="merchant_auth_rejected",
            input_data={"route": "merchant_api", "header_present": authorization is not None},
            decision_reason="missing, malformed or unknown merchant key",
            success=False,
        )
        raise HTTPException(
            status_code=401,
            detail="invalid merchant key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return merchant_id


def _safe_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip operator-only fields from a timeline."""
    out = []
    for event in events:
        detail = event.get("detail") or {}
        out.append({
            "stage": event["stage"],
            "timestamp": event["timestamp"],
            "detail": {k: v for k, v in detail.items() if k in SAFE_DETAIL_KEYS},
        })
    return out


def _safe_approval(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in SAFE_APPROVAL_KEYS}


@router.get("/me")
def me(merchant_id: str = Depends(require_merchant)) -> dict[str, Any]:
    """Identify the merchant behind the key."""
    record = next((m for m in merchants.list_merchants() if m["id"] == merchant_id), None)
    return {"merchant_id": merchant_id, "name": record["name"] if record else None}


@router.get("/summary")
def summary(
    limit: int = Query(default=50, ge=1, le=200),
    merchant_id: str = Depends(require_merchant),
) -> dict[str, Any]:
    """This merchant's live state: counts, active and failing recoveries, pending approvals."""
    active, active_total = recovery_store.list_cases(status="pending", merchant_id=merchant_id, limit=limit)
    escalated_total = recovery_store.list_cases(status="escalated", merchant_id=merchant_id, limit=1)[1]
    settled_total = recovery_store.list_cases(status="settled", merchant_id=merchant_id, limit=1)[1]
    settled_today = recovery_store.get_settled_today(merchant_id=merchant_id)

    failures: list[dict[str, Any]] = []
    candidates, _ = recovery_store.list_cases(status="pending", merchant_id=merchant_id, limit=200)
    for case in candidates:
        failure = recovery_store.latest_event(case["recovery_id"], FAILURE_STAGES)
        if failure is not None:
            failures.append({
                **case,
                "failure_stage": failure["stage"],
                "failure_at": failure["timestamp"],
            })
    failures.sort(key=lambda row: row["failure_at"] or "", reverse=True)

    pending_approvals, pending_total = approvals.list_approvals("pending", merchant_id, None, limit, 0)

    return {
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "counts": {
            "active": active_total,
            "settled": settled_total,
            "escalated": escalated_total,
            "failures_in_progress": len(failures),
            "pending_approvals": pending_total,
            "settled_today": len(settled_today),
        },
        "settled_today_inr": round(sum(float(r["amount_inr"]) for r in settled_today), 2),
        "active": [_with_latest(case) for case in active],
        "failures": failures[:limit],
        "approvals": [_safe_approval(row) for row in pending_approvals],
    }


def _with_latest(case: dict[str, Any]) -> dict[str, Any]:
    latest = recovery_store.latest_event(case["recovery_id"], recovery_store.STAGES)
    return {
        **case,
        "latest_stage": latest["stage"] if latest else None,
        "latest_at": latest["timestamp"] if latest else None,
    }


@router.get("/recoveries")
def list_recoveries(
    status: Literal["pending", "settled", "failed", "escalated"] | None = Query(default=None),
    cause: str | None = Query(default=None, max_length=60),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    merchant_id: str = Depends(require_merchant),
) -> dict[str, Any]:
    """This merchant's recoveries, newest first."""
    items, total = recovery_store.list_cases(
        status=status, merchant_id=merchant_id, cause=cause, limit=limit, offset=offset,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/recoveries/{recovery_id}")
def get_recovery(recovery_id: str, merchant_id: str = Depends(require_merchant)) -> dict[str, Any]:
    """One recovery with its sanitised timeline. 404 for another merchant's recovery."""
    case = recovery_store.get_recovery_status(recovery_id)
    if case is None or case["merchant_id"] != merchant_id:
        raise HTTPException(status_code=404, detail="recovery not found")
    return {
        "recovery": case,
        "timeline": _safe_timeline(recovery_store.get_recovery_timeline(recovery_id)),
    }


# --- the web page ----------------------------------------------------------------

_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def _file(name: str, media_type: str) -> FileResponse:
    response = FileResponse(WEB_DIR / name, media_type=media_type)
    response.headers["Content-Security-Policy"] = _CSP
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@page_router.get("/merchant")
def merchant_page() -> FileResponse:
    """The merchant dashboard page."""
    return _file("index.html", "text/html; charset=utf-8")


@page_router.get("/merchant/app.js")
def merchant_js() -> FileResponse:
    """Page script."""
    return _file("app.js", "text/javascript; charset=utf-8")


@page_router.get("/merchant/app.css")
def merchant_css() -> FileResponse:
    """Page styles."""
    return _file("app.css", "text/css; charset=utf-8")
