"""API request/response schemas.

Pydantic validates in both directions: everything coming in from an operator,
and everything coming out of Apertus. The second direction is the one that
matters for this project, because a model output that fails validation is a
claim we cannot trace and must not be persisted.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy import (
    ConflictStatus,
    ConflictType,
    EventType,
    Severity,
    SourceType,
    VerificationState,
)


class ReportIn(BaseModel):
    """Operator-submitted report. raw_text is stored verbatim and never edited."""

    message: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="und", max_length=8)
    source_type: SourceType = SourceType.HUMAN_REPORT
    location_hint: str | None = Field(default=None, max_length=400)
    reporter_id: str | None = Field(default=None, max_length=64)
    timestamp: datetime | None = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("language")
    @classmethod
    def _lang(cls, v: str) -> str:
        v = (v or "und").strip().lower()
        return v[:8] if v else "und"


class BatchIn(BaseModel):
    reports: list[ReportIn] = Field(min_length=1, max_length=500)


class ReportOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    report_id: str
    ingested_at: datetime
    reported_at: datetime | None
    source_type: str
    source_language: str
    raw_text: str
    location_hint: str | None
    reporter_id: str | None
    processed: bool
    processed_at: datetime | None
    suspected_injection: bool
    injection_signals: list
    segment_count: int


class EvidenceLinkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    link_id: str
    report_id: str
    field_path: str
    reason: str


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    report_id: str
    event_type: str
    description: str | None
    location_text: str | None
    resolved_location: str | None
    people_affected: int | None
    severity: str
    time_reference: str | None
    uncertainties: list
    evidence_confidence: float
    verification_state: str
    independent_source_count: int
    model_used: str
    model_label: str
    prompt_version: str
    guards_applied: list
    created_at: datetime


class EventDetail(EventOut):
    """An event with the full evidence chain.

    The evidence panel in the UI renders exactly this: the structured claim, the
    confidence figures, and every original report behind it in its own language.
    """

    source_language: str
    raw_text: str
    evidence_links: list[EvidenceLinkOut]
    conflicts: list["ConflictOut"] = Field(default_factory=list)


class ConflictOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conflict_id: str
    event_a_id: str
    event_b_id: str
    conflict_type: str
    severity: str
    explanation: str
    evidence: list
    status: str
    resolved_by: str | None
    resolved_at: datetime | None
    resolution_note: str | None
    detected_at: datetime
    detection_rule: str
    model_used: str | None


class ConflictDetail(ConflictOut):
    """Both sides in full, including each original report.

    This is the screen the whole project is arguing for: both claims, side by
    side, with their sources and timestamps, and an explicit UNRESOLVED status
    instead of a confident answer.
    """

    event_a: EventOut
    event_b: EventOut
    report_a: ReportOut
    report_b: ReportOut


class VerifyIn(BaseModel):
    state: VerificationState
    note: str | None = Field(default=None, max_length=2000)
    actor: str | None = Field(default=None, max_length=64)


class ResolveIn(BaseModel):
    action: ConflictStatus
    note: str = Field(min_length=1, max_length=2000)
    actor: str | None = Field(default=None, max_length=64)


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    audit_id: str
    entity_type: str
    entity_id: str
    action: str
    model_used: str | None
    model_version: str | None
    prompt_version: str | None
    input_ref: str | None
    output_summary: dict
    actor: str | None
    detail: dict
    timestamp: datetime


class MetricsOut(BaseModel):
    reports_total: int
    reports_processed: int
    events_total: int
    events_unverified: int
    events_corroborated: int
    events_disputed: int
    events_verified_by_operator: int
    events_rejected: int
    events_critical: int
    people_affected_reported: int
    people_affected_lower_bound: int
    locations_affected: int
    conflicts_unresolved: int
    conflicts_resolved: int
    duplicate_groups: int
    independent_source_max: int
    suspected_injection_reports: int
    model_label: str
    model_used: str


class HealthOut(BaseModel):
    status: str
    version: str
    apertus: dict
    reports: int
    events: int


EventDetail.model_rebuild()