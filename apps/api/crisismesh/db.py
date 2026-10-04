"""Persistence layer.

SQLite by default: zero resident memory, single file, no extension management.
That matters here because the demo box has 16 GB of RAM shared with the model.
Postgres is supported by changing DATABASE_URL and is exercised in compose.

Note on pgvector: we deliberately do NOT use an embedding store. Duplicate
detection is done with lexical similarity (trigram Jaccard) for candidate
generation plus Apertus adjudication for the final call. At CrisisMesh's report
volumes this is both cheaper and more transparent, and it keeps the "no GPU,
no extra services" claim honest. See docs/ARCHITECTURE.md.

Immutability of reports.raw_text is enforced at the ORM level, not by
convention: an UPDATE touching raw_text raises.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _uuid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class Report(Base):
    """An immutable human report. The root of all evidence.

    raw_text is never updated or deleted. Everything CrisisMesh believes is
    derived from a Report and is reachable through an EvidenceLink.
    """

    __tablename__ = "reports"

    report_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                          default=lambda: _uuid("REP"))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=utcnow)
    reported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_type: Mapped[str] = mapped_column(String(32), default="human_report")
    source_language: Mapped[str] = mapped_column(String(8), default="und")
    raw_text: Mapped[str] = mapped_column(Text)
    location_hint: Mapped[str | None] = mapped_column(Text)
    reporter_id: Mapped[str | None] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    text_hash: Mapped[str] = mapped_column(String(64), index=True)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suspected_injection: Mapped[bool] = mapped_column(Boolean, default=False)
    injection_signals: Mapped[list] = mapped_column(JSON, default=list)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)

    events: Mapped[list["Event"]] = relationship(back_populates="report")

    @property
    def raw_text_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()


class Event(Base):
    """A structured situation item extracted by Apertus from a Report.

    An Event is a *claim*, not a fact. Its authority comes entirely from its
    EvidenceLinks and its verification_state, neither of which the model sets.
    """

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                          default=lambda: _uuid("EVT"))
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.report_id"),
                                           index=True)
    segment_index: Mapped[int] = mapped_column(Integer, default=0)

    event_type: Mapped[str] = mapped_column(String(48), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    location_text: Mapped[str | None] = mapped_column(Text, index=True)
    resolved_location: Mapped[str | None] = mapped_column(String(160))
    location_confidence: Mapped[float | None] = mapped_column(Float)
    people_affected: Mapped[int | None] = mapped_column(Integer)
    severity: Mapped[str] = mapped_column(String(16), default="unknown", index=True)
    time_reference: Mapped[str | None] = mapped_column(String(160))
    event_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    uncertainties: Mapped[list] = mapped_column(JSON, default=list)

    # Provenance and confidence are separate by design. Model self-report is
    # not evidence strength; evidence_confidence is derived from independent
    # corroboration, never from the model.
    model_confidence: Mapped[float | None] = mapped_column(Float)
    evidence_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    verification_state: Mapped[str] = mapped_column(String(24),
                                                    default="unverified", index=True)
    independent_source_count: Mapped[int] = mapped_column(Integer, default=1)
    duplicate_group_id: Mapped[str | None] = mapped_column(String(32), index=True)

    model_used: Mapped[str] = mapped_column(String(64))
    model_label: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    grammar_version: Mapped[str] = mapped_column(String(32), default="gbnf_v1")
    extraction_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    guards_applied: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow)

    report: Mapped[Report] = relationship(back_populates="events")
    evidence_links: Mapped[list["EvidenceLink"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class Entity(Base):
    __tablename__ = "entities"

    entity_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                           default=lambda: _uuid("ENT"))
    name: Mapped[str] = mapped_column(String(200), index=True)
    entity_type: Mapped[str] = mapped_column(String(40), default="place")
    language: Mapped[str | None] = mapped_column(String(8))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow)


class EventEntity(Base):
    __tablename__ = "event_entities"

    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"),
                                          primary_key=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.entity_id"),
                                           primary_key=True)
    role: Mapped[str | None] = mapped_column(String(40))


class EvidenceLink(Base):
    """Provenance edge: which report supports which field of which event.

    Every non-null factual field must have at least one of these. This is the
    hard invariant behind the project's core promise, and it is what makes the
    Truth Preservation Score reportable as VALID or INVALID.
    """

    __tablename__ = "evidence_links"

    link_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                         default=lambda: _uuid("EL"))
    event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), index=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.report_id"), index=True)
    field_path: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(40), default="extracted_directly")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow)

    event: Mapped[Event] = relationship(back_populates="evidence_links")


class Conflict(Base):
    """An unresolved contradiction between two claims.

    CrisisMesh never resolves these itself. It surfaces both sides and hands
    the decision to a human operator.
    """

    __tablename__ = "conflicts"

    conflict_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                             default=lambda: _uuid("CFL"))
    event_a_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), index=True)
    event_b_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), index=True)
    conflict_type: Mapped[str] = mapped_column(String(24), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="unresolved", index=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  default=utcnow)
    detection_rule: Mapped[str] = mapped_column(String(64), default="deterministic")
    model_used: Mapped[str | None] = mapped_column(String(64))


class DuplicateGroup(Base):
    """A set of reports judged to be the same underlying observation.

    Grouping matters epistemically: ten forwarded copies of one message are one
    source, not ten. Corroboration counts independent groups, never rows.
    """

    __tablename__ = "duplicate_groups"

    group_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                           default=lambda: _uuid("DUP"))
    method: Mapped[str] = mapped_column(String(24), default="lexical_trigram")
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    member_report_ids: Mapped[list] = mapped_column(JSON, default=list)
    member_event_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=utcnow)


class AuditLog(Base):
    """Append-only record of every transformation and operator action.

    Deliberately stores identifiers and metadata, not full report bodies, so the
    audit trail can be retained without duplicating sensitive content.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[str] = mapped_column(String(32), primary_key=True,
                                          default=lambda: _uuid("ALOG"))
    entity_type: Mapped[str] = mapped_column(String(24), index=True)
    entity_id: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(40), index=True)
    model_used: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    input_ref: Mapped[str | None] = mapped_column(String(64))
    output_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    actor: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                default=utcnow)


Index("ix_events_type_state", Event.event_type, Event.verification_state)
Index("ix_conflicts_status_type", Conflict.status, Conflict.conflict_type)


# ---- immutability guard -----------------------------------------------------

@event.listens_for(Report, "before_update")
def _block_raw_text_mutation(mapper, connection, target) -> None:
    """Refuse any UPDATE that would change raw_text.

    Evidence you can edit is not evidence. This is enforced rather than trusted
    so that a future contributor cannot break the invariant by accident.
    """
    if "raw_text" in target.__dict__:
        from sqlalchemy import inspect as sa_inspect
        hist = sa_inspect(target).attrs.raw_text.history
        if hist.has_changes():
            raise ValueError(
                "Report.raw_text is immutable; it is the root of all evidence."
            )


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        from .config import get_settings
        url = get_settings().database_url
        kwargs = {"future": True}
        if url.startswith("sqlite"):
            from pathlib import Path as _P
            db_path = url.split("///")[-1]
            if db_path and db_path != ":memory:":
                _P(db_path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False,
                                       future=True)
    return _SessionFactory


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def session_scope():
    """Context manager yielding a session with commit/rollback handling."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        s = get_session_factory()()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    return _cm()