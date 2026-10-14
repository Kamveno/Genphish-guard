"""SOAR integration layer for genphish-guard.

Architecture
------------
The zero-trust response module (`zerotrust.py`) assigns a tier (BLOCK /
DEFUSE / DELIVER) to each email.  This module translates that tier into
SOAR incidents and dispatches them to a connector.  Three connectors ship:

    GenericWebhookConnector  — POST JSON to any webhook (Splunk SOAR,
                               Palo Alto XSOAR, Microsoft Sentinel, etc.)
    SplunkSOARConnector      — thin wrapper around the Splunk SOAR REST API
    SimulationConnector      — logs incidents to an in-memory queue;
                               used for demos and the Streamlit dashboard

In the dissertation this module demonstrates the *operational* end of the
pipeline: a detection result does not sit in a CSV — it triggers a
response.

Usage (simulation, no real SOAR needed)
----------------------------------------
    from genphish import soar as SOAR
    incidents = SOAR.incidents_from_scores(scores, tiers, method, source)
    SOAR.log_to_session(incidents)      # write to Streamlit session state
    log = SOAR.get_session_log()        # read it back

Usage (production)
------------------
    connector = SOAR.GenericWebhookConnector("https://soar.example.com/api/v2/incidents")
    for inc in incidents:
        connector.send(inc)

Playbook mapping
----------------
    BLOCK   → isolate sender, block at MTA, create P1 ticket
    DEFUSE  → strip URLs, sandbox attachment, create P2 ticket, notify user
    DELIVER → log for audit trail, no action

The playbook is intentionally minimal so the panel can see the mapping
without wading through real SOAR SDK boilerplate.
"""
from __future__ import annotations
import json, time, uuid
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

# ------------------------------------------------------------------ incident
@dataclass
class Incident:
    """One detection event, ready to hand to a SOAR platform."""
    id:           str
    timestamp:    float
    tier:         str          # BLOCK | DEFUSE | DELIVER
    confidence:   float        # raw model score
    threshold:    float
    cal_method:   str          # stale | dm | em | self | oracle
    source_det:   str          # which detector produced the score
    severity:     str          # P1 | P2 | P3 (derived from tier)
    raw_features: dict = field(default_factory=dict)

    # derived
    @property
    def priority(self) -> int:
        return {"BLOCK": 1, "DEFUSE": 2, "DELIVER": 3}.get(self.tier, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["priority"] = self.priority
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _severity(tier: str) -> str:
    return {"BLOCK": "P1-Critical", "DEFUSE": "P2-High", "DELIVER": "P3-Info"}.get(tier, "P3-Info")


def incidents_from_scores(
    scores: np.ndarray,
    tiers: list[str],
    cal_method: str,
    source_detector: str,
    threshold: float = 0.5,
    raw_features: list[dict] | None = None,
) -> list[Incident]:
    """Convert a batch of scored emails into SOAR incidents.

    Only BLOCK and DEFUSE rows create incidents (DELIVER is audit-only).
    """
    incidents = []
    for i, (score, tier) in enumerate(zip(scores, tiers)):
        if tier == "DELIVER":
            continue
        feat = (raw_features[i] if raw_features and i < len(raw_features) else {})
        incidents.append(Incident(
            id=str(uuid.uuid4())[:8].upper(),
            timestamp=time.time(),
            tier=tier,
            confidence=float(score),
            threshold=float(threshold),
            cal_method=cal_method,
            source_det=source_detector,
            severity=_severity(tier),
            raw_features=feat,
        ))
    return incidents


# ------------------------------------------------------------------ playbook
_PLAYBOOK: dict[str, list[dict]] = {
    "BLOCK": [
        {"action": "block_sender_at_mta",
         "detail": "Add sender domain to MTA blocklist immediately",
         "status": "executed"},
        {"action": "create_ticket",
         "detail": "P1 ticket in ticketing system — SOC review required",
         "status": "executed"},
        {"action": "notify_soc",
         "detail": "Push alert to SOC Slack / PagerDuty",
         "status": "executed"},
        {"action": "quarantine_copies",
         "detail": "Move any delivered copies to quarantine mailbox",
         "status": "executed"},
    ],
    "DEFUSE": [
        {"action": "strip_urls",
         "detail": "Replace hyperlinks with [REDACTED-PHISHING-LINK]",
         "status": "executed"},
        {"action": "sandbox_attachments",
         "detail": "Submit attachments to sandbox analysis queue",
         "status": "executed"},
        {"action": "notify_recipient",
         "detail": "Send user awareness notice before delivering defused copy",
         "status": "executed"},
        {"action": "create_ticket",
         "detail": "P2 ticket — review in next analyst batch",
         "status": "executed"},
    ],
    "DELIVER": [
        {"action": "audit_log",
         "detail": "Record to immutable audit trail; no action taken",
         "status": "skipped"},
    ],
}


def playbook_response(incident: Incident) -> list[dict]:
    """Return the playbook steps for this incident's tier."""
    return _PLAYBOOK.get(incident.tier, _PLAYBOOK["DELIVER"])


# ------------------------------------------------------------------ connectors
class SimulationConnector:
    """In-memory connector used for tests and the Streamlit demo."""
    def __init__(self) -> None:
        self._queue: list[Incident] = []

    def send(self, incident: Incident) -> dict:
        self._queue.append(incident)
        return {"status": "queued", "id": incident.id}

    def flush(self) -> list[Incident]:
        out, self._queue = self._queue[:], []
        return out

    def queue(self) -> list[Incident]:
        return list(self._queue)


class GenericWebhookConnector:
    """POST each incident as JSON to any webhook endpoint.

    Works with Splunk SOAR, Palo Alto XSOAR, Microsoft Sentinel Logic Apps,
    or any custom endpoint.  Requires `requests`; silently no-ops if missing.
    """
    def __init__(self, endpoint: str, headers: dict | None = None,
                 timeout: int = 10) -> None:
        self.endpoint = endpoint
        self.headers  = headers or {"Content-Type": "application/json"}
        self.timeout  = timeout

    def send(self, incident: Incident) -> dict:
        try:
            import requests
            resp = requests.post(
                self.endpoint,
                data=incident.to_json(),
                headers=self.headers,
                timeout=self.timeout,
            )
            return {"status": resp.status_code, "id": incident.id}
        except ImportError:
            return {"status": "no_requests_library", "id": incident.id}
        except Exception as e:
            return {"status": "error", "error": str(e), "id": incident.id}


class SplunkSOARConnector:
    """Splunk SOAR (Phantom) REST API connector.

    Example
    -------
        conn = SplunkSOARConnector(
            host="https://splunk-soar.example.com",
            auth_token="YOUR_TOKEN",
        )
        conn.send(incident)
    """
    def __init__(self, host: str, auth_token: str, container_label: str = "phishing") -> None:
        self.host = host.rstrip("/")
        self.auth_token = auth_token
        self.container_label = container_label

    def _headers(self) -> dict:
        return {
            "ph-auth-token": self.auth_token,
            "Content-Type": "application/json",
        }

    def send(self, incident: Incident) -> dict:
        try:
            import requests
            payload = {
                "name": f"genphish-guard | {incident.tier} | {incident.id}",
                "label": self.container_label,
                "severity": incident.severity.split("-")[0].lower(),
                "status": "new",
                "description": (
                    f"LLM-generated phishing detected by genphish-guard.\n"
                    f"Confidence: {incident.confidence:.3f}  "
                    f"Threshold: {incident.threshold:.3f}  "
                    f"Calibration: {incident.cal_method}  "
                    f"Detector: {incident.source_det}"
                ),
                "custom_fields": incident.to_dict(),
            }
            resp = requests.post(
                f"{self.host}/rest/container",
                json=payload,
                headers=self._headers(),
                verify=True,
                timeout=15,
            )
            return {"status": resp.status_code, "id": incident.id,
                    "container_id": resp.json().get("id")}
        except ImportError:
            return {"status": "no_requests_library", "id": incident.id}
        except Exception as e:
            return {"status": "error", "error": str(e), "id": incident.id}


# ------------------------------------------------------------------ session helpers
# Used by the Streamlit dashboard to persist the incident log across reruns.
_SIM_CONNECTOR = SimulationConnector()


def log_to_session(incidents: list[Incident]) -> None:
    """Write incidents to session state if Streamlit is available,
    else fall back to the simulation connector."""
    try:
        import streamlit as st
        if "soar_log" not in st.session_state:
            st.session_state["soar_log"] = []
        st.session_state["soar_log"].extend(incidents)
    except Exception:
        for inc in incidents:
            _SIM_CONNECTOR.send(inc)


def get_session_log() -> list[Incident]:
    try:
        import streamlit as st
        return st.session_state.get("soar_log", [])
    except Exception:
        return _SIM_CONNECTOR.queue()


def clear_session_log() -> None:
    try:
        import streamlit as st
        st.session_state["soar_log"] = []
    except Exception:
        _SIM_CONNECTOR.flush()
