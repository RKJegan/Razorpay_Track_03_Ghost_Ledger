"""
Ghost Ledger v3 — YAML playbooks (B1).

One playbook per root cause lives in ``playbooks/<cause>.yaml``. A playbook
says *what the deterministic router may do* for that cause. It never contains
an amount, an approval, a gateway secret, or a model prompt.

:class:`PlaybookLoader`
    Loads every playbook, validates the whole set, and refuses to run with an
    invalid set. Hot reload: when a file changes on disk, the set is reloaded
    on the next :meth:`PlaybookLoader.get`. If the edited set is invalid, the
    last good set keeps running and the error is kept in ``last_error``.

:class:`PlaybookExecutor`
    Lives in :mod:`strategies.executor`. It turns a routed plan into events and
    scheduled work. It is not defined here, so this module stays free of side
    effects.

Schema (all keys are required unless marked optional)::

    cause: insufficient_funds          # one of config.CAUSE_BUCKETS
    version: 1                         # positive integer, bump on every edit
    description: "..."                 # free text, shown in the dry run
    route: create_link                 # create_link | dunning_only
    retry:                             # optional; default timing_rule: none
      timing_rule: cycle_aware_funds   # none | cycle_aware_funds | peak_avoidance
    failover:                          # optional; default disabled
      enabled: true
      candidates: [upi, netbanking]
    method_suggestion:                 # optional; disabled when absent
      after_card_failures: 2
      candidates: [upi, netbanking]
    dunning:                           # optional; no touches when absent
      touches:
        - {offset_minutes: 0, channel: email, template: funds_first}
    templates:                         # required when dunning.touches is set
      funds_first: "Hello, your payment of INR {amount_inr} for {txn_id} did not go through. {next_step}"

Trust boundary: this module only reads YAML. Nothing here calls an LLM.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import config
from strategies.timing import TIMING_RULES

logger = logging.getLogger(__name__)

ROUTES: tuple[str, ...] = ("create_link", "dunning_only")
CHANNELS: tuple[str, ...] = ("email", "sms", "whatsapp")
PAYMENT_METHODS: tuple[str, ...] = ("card", "upi", "netbanking", "wallet", "emi")
TEMPLATE_FIELDS: frozenset[str] = frozenset({"amount_inr", "txn_id", "next_step"})
TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "cause", "version", "description", "route", "retry", "failover",
    "method_suggestion", "dunning", "templates",
})
MAX_TOUCHES = 6


class PlaybookError(ValueError):
    """Raised when a playbook file or the playbook set is invalid."""


@dataclass(frozen=True)
class Touch:
    """One dunning reminder: when (minutes after the failure), how, and which template."""

    touch_no: int
    offset_minutes: int
    channel: str
    template: str


@dataclass(frozen=True)
class Playbook:
    """A validated playbook for one root cause."""

    cause: str
    version: int
    description: str
    route: str
    timing_rule: str
    failover_enabled: bool
    failover_candidates: tuple[str, ...]
    method_candidates: tuple[str, ...]
    method_after_card_failures: int | None
    touches: tuple[Touch, ...]
    templates: dict[str, str] = field(default_factory=dict)
    source: str = ""


def _fail(source: str, message: str) -> PlaybookError:
    return PlaybookError(f"{source}: {message}")


def _as_mapping(value: Any, source: str, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _fail(source, f"{where} must be a mapping")
    return value


def _methods(value: Any, source: str, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _fail(source, f"{where} must be a list of method names")
    for name in value:
        if name not in PAYMENT_METHODS:
            raise _fail(source, f"{where} has unknown method {name!r}; allowed: {PAYMENT_METHODS}")
    return tuple(value)


def _parse(path: Path) -> Playbook:
    """Parse and validate one playbook file. Raises PlaybookError with the file name."""
    source = path.name
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise _fail(source, f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise _fail(source, "top level must be a mapping")

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise _fail(source, f"unknown keys {sorted(unknown)}")

    cause = raw.get("cause")
    if cause not in config.CAUSE_BUCKETS:
        raise _fail(source, f"cause {cause!r} is not one of {config.CAUSE_BUCKETS}")
    if path.stem != cause:
        raise _fail(source, f"file name must match cause ({cause}.yaml)")

    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise _fail(source, "version must be a positive integer")

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise _fail(source, "description must be text")

    route = raw.get("route")
    if route not in ROUTES:
        raise _fail(source, f"route must be one of {ROUTES}")

    retry = _as_mapping(raw.get("retry"), source, "retry")
    timing_rule = retry.get("timing_rule", "none")
    if timing_rule not in TIMING_RULES:
        raise _fail(source, f"retry.timing_rule must be one of {tuple(TIMING_RULES)}")
    if set(retry) - {"timing_rule"}:
        raise _fail(source, f"retry has unknown keys {sorted(set(retry) - {'timing_rule'})}")

    failover = _as_mapping(raw.get("failover"), source, "failover")
    if set(failover) - {"enabled", "candidates"}:
        raise _fail(source, "failover has unknown keys")
    failover_enabled = bool(failover.get("enabled", False))
    failover_candidates = _methods(failover.get("candidates"), source, "failover.candidates")
    if failover_enabled and not failover_candidates:
        raise _fail(source, "failover.enabled needs failover.candidates")

    suggestion = _as_mapping(raw.get("method_suggestion"), source, "method_suggestion")
    if set(suggestion) - {"after_card_failures", "candidates"}:
        raise _fail(source, "method_suggestion has unknown keys")
    method_after = suggestion.get("after_card_failures")
    if method_after is not None and (
        not isinstance(method_after, int) or isinstance(method_after, bool) or method_after < 1
    ):
        raise _fail(source, "method_suggestion.after_card_failures must be a positive integer")
    method_candidates = _methods(suggestion.get("candidates"), source, "method_suggestion.candidates")
    if method_after is not None and not method_candidates:
        raise _fail(source, "method_suggestion needs candidates")

    dunning = _as_mapping(raw.get("dunning"), source, "dunning")
    if set(dunning) - {"touches"}:
        raise _fail(source, "dunning has unknown keys")
    templates = _as_mapping(raw.get("templates"), source, "templates")
    for name, text in templates.items():
        if not isinstance(text, str):
            raise _fail(source, f"template {name!r} must be text")
        try:
            fields = {f for _, f, _, _ in string.Formatter().parse(text) if f}
        except ValueError as exc:
            raise _fail(source, f"template {name!r} is malformed: {exc}") from exc
        if not fields <= TEMPLATE_FIELDS:
            raise _fail(source, f"template {name!r} uses unknown fields {sorted(fields - TEMPLATE_FIELDS)}")

    raw_touches = dunning.get("touches") or []
    if not isinstance(raw_touches, list):
        raise _fail(source, "dunning.touches must be a list")
    if len(raw_touches) > MAX_TOUCHES:
        raise _fail(source, f"at most {MAX_TOUCHES} dunning touches per cause")
    touches: list[Touch] = []
    last_offset = -1
    for index, item in enumerate(raw_touches, start=1):
        if not isinstance(item, dict) or set(item) != {"offset_minutes", "channel", "template"}:
            raise _fail(source, f"touch {index} needs exactly offset_minutes, channel, template")
        offset = item["offset_minutes"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise _fail(source, f"touch {index}: offset_minutes must be an integer >= 0")
        if offset < last_offset:
            raise _fail(source, f"touch {index}: offsets must not decrease")
        last_offset = offset
        if item["channel"] not in CHANNELS:
            raise _fail(source, f"touch {index}: channel must be one of {CHANNELS}")
        if item["template"] not in templates:
            raise _fail(source, f"touch {index}: template {item['template']!r} is not defined")
        touches.append(Touch(index, offset, item["channel"], item["template"]))
    if touches and not templates:
        raise _fail(source, "dunning.touches needs templates")

    return Playbook(
        cause=cause,
        version=version,
        description=description,
        route=route,
        timing_rule=timing_rule,
        failover_enabled=failover_enabled,
        failover_candidates=failover_candidates,
        method_candidates=method_candidates,
        method_after_card_failures=method_after,
        touches=tuple(touches),
        templates={k: str(v) for k, v in templates.items()},
        source=source,
    )


def load_playbook_set(directory: Path) -> dict[str, Playbook]:
    """
    Load and validate every playbook in ``directory``.

    The set must cover every cause in ``config.CAUSE_BUCKETS`` exactly once.
    All errors are collected and raised together, so one run shows everything.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise PlaybookError(f"playbook directory not found: {directory}")

    books: dict[str, Playbook] = {}
    errors: list[str] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            book = _parse(path)
        except PlaybookError as exc:
            errors.append(str(exc))
            continue
        if book.cause in books:
            errors.append(f"{path.name}: cause {book.cause!r} is defined twice")
            continue
        books[book.cause] = book

    missing = [c for c in config.CAUSE_BUCKETS if c not in books]
    if missing:
        errors.append(f"no playbook for causes {missing}")
    if errors:
        raise PlaybookError("invalid playbook set: " + "; ".join(errors))
    return books


class PlaybookLoader:
    """Holds the validated playbook set and reloads it when files change."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory or config.PLAYBOOK_DIR)
        self._books: dict[str, Playbook] = {}
        self._signature: tuple[Any, ...] | None = None
        self.last_error: str | None = None

    def _current_signature(self) -> tuple[Any, ...]:
        if not self.directory.is_dir():
            return ()
        return tuple(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(self.directory.glob("*.yaml"))
        )

    def load(self) -> dict[str, Playbook]:
        """Load now. Raises PlaybookError if the set is invalid (nothing is replaced)."""
        books = load_playbook_set(self.directory)
        self._books = books
        self._signature = self._current_signature()
        self.last_error = None
        return dict(books)

    def reload_if_changed(self) -> None:
        """Reload when a file was added, removed, or edited. Keep the last good set on error."""
        signature = self._current_signature()
        if signature == self._signature and self._books:
            return
        try:
            self.load()
            logger.info("playbooks loaded: %d causes", len(self._books))
        except PlaybookError as exc:
            self.last_error = str(exc)
            self._signature = signature  # do not retry until the files change again
            if not self._books:
                raise
            logger.error("playbook reload rejected, keeping last good set: %s", exc)

    def get(self, cause: str) -> Playbook:
        """Return the playbook for ``cause`` (reloading first if the files changed)."""
        if not self._books:
            self.load()
        self.reload_if_changed()
        try:
            return self._books[cause]
        except KeyError as exc:
            raise PlaybookError(f"no playbook for cause {cause!r}") from exc

    def all(self) -> dict[str, Playbook]:
        """Return the current validated set."""
        if not self._books:
            self.load()
        self.reload_if_changed()
        return dict(self._books)


_default: PlaybookLoader | None = None


def default_loader() -> PlaybookLoader:
    """Return the process-wide loader (created on first use)."""
    global _default
    if _default is None:
        _default = PlaybookLoader()
    return _default
