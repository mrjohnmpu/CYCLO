#!/usr/bin/env python3
"""Classify peptide cyclization from Bio Parser component-graph records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx
import pandas as pd


DEFAULT_API_BASE_URL = os.environ.get("PEPTIDE_PARSER_API_URL", "http://127.0.0.1:8000")
DEFAULT_SMILES_COLUMN = "SMILES"
DEFAULT_BATCH_SIZE = 50
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_JOB_TIMEOUT = 3600.0
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
COMPONENT_GRAPH_IDENTIFIER = "covalent-unit-multigraph"
TARGET_LABELS = ("head_to_tail", "sc_to_tail", "disulfide_bridge")
PREDICTION_LABELS = TARGET_LABELS + (
    "linear",
    "other",
    "ambiguous",
    "failed",
)
OUTPUT_COLUMNS = [
    "source_index",
    "source_id",
    "input_smiles",
    "expected_cyclization",
    "api_record_id",
    "predicted_cyclization",
    "prediction_status",
    "candidate_labels_json",
    "component_cycle_rank",
    "closure_count",
    "is_multicyclic",
    "evidence_json",
    "api_processing_status",
    "api_error_code",
    "api_error_message",
    "prediction_error_code",
    "prediction_error_message",
]
TRANSIENT_HTTP_STATUSES = {429, 502, 503, 504}
SUCCESS_JOB_STATUSES = {"completed", "complete", "succeeded", "success", "done"}
FAILED_JOB_STATUSES = {"failed", "error", "cancelled", "canceled"}


class BioParserError(RuntimeError):
    """Base error for service or contract failures."""


class BatchValidationError(BioParserError):
    """The service rejected a batch request with HTTP 422."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ExpiredBatchError(BioParserError):
    """A submitted batch disappeared before its result was collected."""


@dataclass
class ClassificationResult:
    predicted_cyclization: str
    prediction_status: str
    candidate_labels: list[str] = field(default_factory=list)
    component_cycle_rank: int = 0
    closure_count: int = 0
    is_multicyclic: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error_code: str = ""
    error_message: str = ""


@dataclass
class BatchExecutionResult:
    outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_shared: int = 0

    def merge(self, other: "BatchExecutionResult") -> None:
        self.outcomes.update(other.outcomes)
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.cache_shared += other.cache_shared


@dataclass(frozen=True)
class InputRow:
    source_index: int
    source_id: str
    smiles: str
    expected_cyclization: str
    api_record_id: str


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def api_record_id(smiles: str) -> str:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    return f"smiles:{digest}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _failed_classification(code: str, message: str) -> ClassificationResult:
    return ClassificationResult(
        predicted_cyclization="failed",
        prediction_status="failed",
        error_code=code,
        error_message=message,
    )


def _component_graph(record: Mapping[str, Any]) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for assertion in _list(record.get("structural_feature_assertions")):
        assertion_dict = _dict(assertion)
        feature_ref = _dict(assertion_dict.get("feature_ref"))
        if feature_ref.get("identifier") == COMPONENT_GRAPH_IDENTIFIER:
            matches.append(assertion_dict)
    if len(matches) != 1:
        return None
    value = matches[0].get("value")
    return value if isinstance(value, dict) else None


def _connection_port_roles(
    connection: Mapping[str, Any],
    ports_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    roles: list[str] = []
    for endpoint_name in ("endpoint_1", "endpoint_2"):
        endpoint = _dict(connection.get(endpoint_name))
        port_id = normalize_text(endpoint.get("port_id"))
        role = normalize_text(_dict(ports_by_id.get(port_id)).get("role"))
        if role:
            roles.append(role)
    return roles


def _edge_component_index(
    edge: Mapping[str, Any],
    cyclic_components: Sequence[tuple[int, set[str]]],
) -> int | None:
    unit_ids = set(_string_list(edge.get("unit_ids")))
    if not unit_ids:
        return None
    for component_index, component_units in cyclic_components:
        if unit_ids.issubset(component_units):
            return component_index
    return None


def _group_component_index(
    group: Mapping[str, Any],
    cyclic_components: Sequence[tuple[int, set[str]]],
) -> int | None:
    unit_ids = set(_string_list(group.get("connector_unit_ids")))
    for endpoint in _list(group.get("polymer_endpoints")):
        unit_id = normalize_text(_dict(endpoint).get("unit_id"))
        if unit_id:
            unit_ids.add(unit_id)
    if not unit_ids:
        return None
    for component_index, component_units in cyclic_components:
        if unit_ids.issubset(component_units):
            return component_index
    return None


def _polymer_endpoint_port_role(
    endpoint: Mapping[str, Any],
    edges_by_id: Mapping[str, Mapping[str, Any]],
    connections_by_id: Mapping[str, Mapping[str, Any]],
    ports_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    physical_edge_id = normalize_text(endpoint.get("physical_edge_id"))
    unit_id = normalize_text(endpoint.get("unit_id"))
    edge = _dict(edges_by_id.get(physical_edge_id))
    matches: list[dict[str, str]] = []
    for connection_id in _string_list(edge.get("connection_ids")):
        connection = _dict(connections_by_id.get(connection_id))
        for endpoint_name in ("endpoint_1", "endpoint_2"):
            connection_endpoint = _dict(connection.get(endpoint_name))
            if normalize_text(connection_endpoint.get("unit_id")) != unit_id:
                continue
            port_id = normalize_text(connection_endpoint.get("port_id"))
            role = normalize_text(_dict(ports_by_id.get(port_id)).get("role"))
            matches.append(
                {
                    "connection_id": connection_id,
                    "port_id": port_id,
                    "role": role,
                }
            )
    roles = sorted({match["role"] for match in matches if match["role"]})
    return {
        "physical_edge_id": physical_edge_id,
        "unit_id": unit_id,
        "roles": roles,
        "matches": matches,
    }


def classify_component_graph(parse_outcome: Mapping[str, Any]) -> ClassificationResult:
    """Classify one Bio Parser ParseOutcome from its component graph."""

    if not isinstance(parse_outcome, Mapping):
        return _failed_classification(
            "invalid_parse_outcome", "Parse outcome must be a mapping"
        )

    outcome_error = _dict(parse_outcome.get("error"))
    record = parse_outcome.get("record")
    if not isinstance(record, Mapping):
        return _failed_classification(
            normalize_text(outcome_error.get("code")) or "missing_record",
            normalize_text(outcome_error.get("message"))
            or "Parse outcome does not contain a record",
        )

    processing_status = normalize_text(record.get("processing_status"))
    if processing_status != "success":
        return _failed_classification(
            normalize_text(outcome_error.get("code")) or "parse_not_successful",
            normalize_text(outcome_error.get("message"))
            or f"processing_status={processing_status or 'missing'}",
        )

    graph = _component_graph(record)
    if graph is None:
        return _failed_classification(
            "missing_component_graph",
            f"Expected exactly one {COMPONENT_GRAPH_IDENTIFIER!r} assertion",
        )

    components = [_dict(item) for item in _list(graph.get("components"))]
    if not components:
        return _failed_classification(
            "invalid_component_graph", "Component graph has no components"
        )

    cyclic_components: list[tuple[int, set[str]]] = []
    component_cycle_rank = 0
    component_summaries: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        cycle_rank = _nonnegative_int(component.get("cycle_rank"))
        unit_ids = set(_string_list(component.get("unit_ids")))
        component_cycle_rank += cycle_rank
        component_summaries.append(
            {
                "component_index": index,
                "cycle_rank": cycle_rank,
                "unit_count": len(unit_ids),
            }
        )
        if cycle_rank > 0:
            cyclic_components.append((index, unit_ids))

    graph_evidence = {
        "kind": "component_graph_summary",
        "schema_version": normalize_text(graph.get("schema_version")),
        "scope": normalize_text(graph.get("scope")),
        "component_cycle_rank": component_cycle_rank,
        "components": component_summaries,
    }
    if component_cycle_rank == 0:
        return ClassificationResult(
            predicted_cyclization="linear",
            prediction_status="success",
            component_cycle_rank=0,
            closure_count=0,
            is_multicyclic=False,
            evidence=[graph_evidence],
        )

    physical_edges = [_dict(item) for item in _list(graph.get("physical_edges"))]
    connector_groups = [_dict(item) for item in _list(graph.get("connector_groups"))]
    connections = [
        _dict(item) for item in _list(record.get("covalent_connections"))
    ]
    ports = [_dict(item) for item in _list(record.get("attachment_ports"))]
    edges_by_id = {
        normalize_text(edge.get("physical_edge_id")): edge
        for edge in physical_edges
        if normalize_text(edge.get("physical_edge_id"))
    }
    connections_by_id = {
        normalize_text(connection.get("connection_id")): connection
        for connection in connections
        if normalize_text(connection.get("connection_id"))
    }
    ports_by_id = {
        normalize_text(port.get("port_id")): port
        for port in ports
        if normalize_text(port.get("port_id"))
    }

    closure_evidence: list[dict[str, Any]] = []
    unresolved_evidence: list[dict[str, Any]] = []
    ambiguous_candidates: set[str] = set()
    seen_closures: set[tuple[str, str]] = set()

    for edge in sorted(
        physical_edges,
        key=lambda item: normalize_text(item.get("physical_edge_id")),
    ):
        component_index = _edge_component_index(edge, cyclic_components)
        if component_index is None:
            continue
        physical_edge_id = normalize_text(edge.get("physical_edge_id"))
        edge_roles = set(_string_list(edge.get("roles")))
        for connection_id in _string_list(edge.get("connection_ids")):
            connection = _dict(connections_by_id.get(connection_id))
            roles = set(edge_roles)
            connection_role = normalize_text(connection.get("role"))
            if connection_role:
                roles.add(connection_role)
            chemistry = normalize_text(connection.get("connection_chemistry"))
            connection_type = normalize_text(connection.get("connection_type"))
            port_roles = _connection_port_roles(connection, ports_by_id)
            port_role_set = set(port_roles)

            if "cyclization" in roles:
                exact_head_to_tail = (
                    chemistry == "amide"
                    and connection_type == "peptide_bond"
                    and {
                        "polymerization_acyl",
                        "polymerization_nucleophile",
                    }.issubset(port_role_set)
                )
                if exact_head_to_tail:
                    key = ("head_to_tail", connection_id)
                    if key not in seen_closures:
                        seen_closures.add(key)
                        closure_evidence.append(
                            {
                                "kind": "closure",
                                "label": "head_to_tail",
                                "component_index": component_index,
                                "physical_edge_id": physical_edge_id,
                                "connection_id": connection_id,
                                "roles": sorted(roles),
                                "connection_chemistry": chemistry,
                                "connection_type": connection_type,
                                "port_roles": sorted(port_role_set),
                            }
                        )
                elif not port_roles or len(port_role_set) < 2:
                    ambiguous_candidates.add("head_to_tail")
                    unresolved_evidence.append(
                        {
                            "kind": "unresolved_cyclization_edge",
                            "component_index": component_index,
                            "physical_edge_id": physical_edge_id,
                            "connection_id": connection_id,
                            "connection_chemistry": chemistry,
                            "connection_type": connection_type,
                            "port_roles": sorted(port_role_set),
                        }
                    )

            is_disulfide = (
                "crosslink" in roles
                and (chemistry == "disulfide" or connection_type == "disulfide")
            )
            if is_disulfide:
                key = ("disulfide_bridge", connection_id)
                if key not in seen_closures:
                    seen_closures.add(key)
                    closure_evidence.append(
                        {
                            "kind": "closure",
                            "label": "disulfide_bridge",
                            "component_index": component_index,
                            "physical_edge_id": physical_edge_id,
                            "connection_id": connection_id,
                            "roles": sorted(roles),
                            "connection_chemistry": chemistry,
                            "connection_type": connection_type,
                            "port_roles": sorted(port_role_set),
                        }
                    )

    for group_index, group in enumerate(connector_groups):
        if group.get("joins_polymer_ports") is not True:
            continue
        component_index = _group_component_index(group, cyclic_components)
        if component_index is None:
            continue
        endpoint_details = [
            _polymer_endpoint_port_role(
                _dict(endpoint),
                edges_by_id,
                connections_by_id,
                ports_by_id,
            )
            for endpoint in _list(group.get("polymer_endpoints"))
        ]
        endpoint_roles = [
            detail["roles"][0]
            for detail in endpoint_details
            if len(detail["roles"]) == 1
        ]
        all_endpoints_resolved = (
            len(endpoint_details) >= 2
            and len(endpoint_roles) == len(endpoint_details)
        )
        role_set = set(endpoint_roles)
        exact_sc_to_tail = (
            all_endpoints_resolved
            and role_set == {"crosslink", "polymerization_acyl"}
        )
        if exact_sc_to_tail:
            group_key = normalize_text(group.get("connector_group_id")) or str(
                group_index
            )
            key = ("sc_to_tail", group_key)
            if key not in seen_closures:
                seen_closures.add(key)
                closure_evidence.append(
                    {
                        "kind": "closure",
                        "label": "sc_to_tail",
                        "component_index": component_index,
                        "connector_group_index": group_index,
                        "boundary_physical_edge_ids": _string_list(
                            group.get("boundary_physical_edge_ids")
                        ),
                        "connector_unit_ids": _string_list(
                            group.get("connector_unit_ids")
                        ),
                        "endpoint_roles": sorted(role_set),
                        "polymer_endpoints": endpoint_details,
                    }
                )
        elif not all_endpoints_resolved and (
            not role_set
            or role_set.intersection({"crosslink", "polymerization_acyl"})
        ):
            ambiguous_candidates.add("sc_to_tail")
            unresolved_evidence.append(
                {
                    "kind": "unresolved_connector_group",
                    "component_index": component_index,
                    "connector_group_index": group_index,
                    "boundary_physical_edge_ids": _string_list(
                        group.get("boundary_physical_edge_ids")
                    ),
                    "endpoint_roles": sorted(role_set),
                    "polymer_endpoints": endpoint_details,
                }
            )

    exact_labels = {
        normalize_text(item.get("label")) for item in closure_evidence
    }
    exact_labels.discard("")
    candidate_labels = [
        label
        for label in TARGET_LABELS
        if label in exact_labels or label in ambiguous_candidates
    ]
    closure_count = component_cycle_rank
    is_multicyclic = closure_count > 1
    evidence = [graph_evidence, *closure_evidence, *unresolved_evidence]

    unresolved_cycle_count = max(0, component_cycle_rank - len(closure_evidence))
    if (
        len(exact_labels) > 1
        or unresolved_evidence
        or (exact_labels and unresolved_cycle_count > 0)
    ):
        if unresolved_cycle_count > 0:
            evidence.append(
                {
                    "kind": "unresolved_cycle_count",
                    "count": unresolved_cycle_count,
                }
            )
        return ClassificationResult(
            predicted_cyclization="ambiguous",
            prediction_status="ambiguous",
            candidate_labels=candidate_labels,
            component_cycle_rank=component_cycle_rank,
            closure_count=closure_count,
            is_multicyclic=is_multicyclic,
            evidence=evidence,
            error_code="ambiguous_component_graph",
            error_message="Component graph contains conflicting or incomplete closure evidence",
        )

    if len(exact_labels) == 1:
        predicted = next(iter(exact_labels))
        return ClassificationResult(
            predicted_cyclization=predicted,
            prediction_status="success",
            candidate_labels=[predicted],
            component_cycle_rank=component_cycle_rank,
            closure_count=closure_count,
            is_multicyclic=is_multicyclic,
            evidence=evidence,
        )

    return ClassificationResult(
        predicted_cyclization="other",
        prediction_status="success",
        component_cycle_rank=component_cycle_rank,
        closure_count=closure_count,
        is_multicyclic=is_multicyclic,
        evidence=evidence,
    )


def synthetic_failed_outcome(
    record_id: str, code: str, message: str
) -> dict[str, Any]:
    return {
        "contract_id": "parse-outcome",
        "contract_version": "1.0.0",
        "record_id": record_id,
        "record": None,
        "error": {
            "code": code,
            "message": message,
            "stage": "client",
            "cause_type": "BioParserClient",
        },
    }


class BioParserClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float,
        job_timeout: float,
        poll_interval: float,
        max_retries: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.job_timeout = job_timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"Accept": "application/json"},
            timeout=httpx.Timeout(request_timeout),
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "BioParserClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any | None = None,
    ) -> httpx.Response:
        last_error = "request failed"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(method, url, json=json_body)
            except httpx.RequestError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                time.sleep(min(60.0, 5.0 * (3**attempt)))
                continue
            if response.status_code not in TRANSIENT_HTTP_STATUSES:
                return response
            last_error = (
                f"HTTP {response.status_code}: {response.text[:500].strip()}"
            )
            if attempt >= self.max_retries:
                break
            time.sleep(min(60.0, 5.0 * (3**attempt)))
        raise BioParserError(
            f"{method} {url} failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _json(response: httpx.Response, context: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BioParserError(
                f"{context} returned invalid JSON: {response.text[:500]}"
            ) from exc
        if not isinstance(payload, dict):
            raise BioParserError(f"{context} returned a non-object JSON value")
        return payload

    def preflight(self, prefer: str) -> dict[str, Any]:
        health_response = self._request("GET", "/healthz")
        if health_response.status_code != 200:
            raise BioParserError(
                f"Health check failed with HTTP {health_response.status_code}: "
                f"{health_response.text[:500]}"
            )
        capabilities_response = self._request("GET", "/api/v1/capabilities")
        if capabilities_response.status_code != 200:
            raise BioParserError(
                "Capabilities request failed with HTTP "
                f"{capabilities_response.status_code}: "
                f"{capabilities_response.text[:500]}"
            )
        capabilities = self._json(capabilities_response, "Capabilities endpoint")
        mode = _dict(_dict(capabilities.get("modes")).get(prefer))
        if mode.get("available") is not True:
            reason = normalize_text(mode.get("reason")) or "unknown reason"
            raise BioParserError(f"Bio Parser mode {prefer!r} is unavailable: {reason}")
        return {
            "health": health_response.text[:500],
            "parser_version": normalize_text(capabilities.get("parser_version")),
            "resource_set_id": normalize_text(capabilities.get("resource_set_id")),
            "default_prefer": normalize_text(capabilities.get("default_prefer")),
            "mode": prefer,
        }

    def _submit(self, items: Sequence[dict[str, str]]) -> dict[str, Any]:
        response = self._request("POST", "/api/v1/batches", json_body={"items": items})
        if response.status_code == 422:
            raise BatchValidationError(response.text[:2000])
        if response.status_code != 202:
            raise BioParserError(
                f"Batch submission failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )
        return self._json(response, "Batch submission")

    def _collect_submitted_batch(
        self, submission: Mapping[str, Any]
    ) -> BatchExecutionResult:
        job_id = normalize_text(submission.get("job_id"))
        if not job_id:
            raise BioParserError("Batch submission did not return job_id")
        status_url = normalize_text(submission.get("status_url")) or (
            f"/api/v1/batches/{job_id}"
        )
        result_url = normalize_text(submission.get("result_url")) or (
            f"/api/v1/batches/{job_id}/result"
        )
        deadline = time.monotonic() + self.job_timeout
        last_status: dict[str, Any] = dict(submission)

        while time.monotonic() < deadline:
            response = self._request("GET", status_url)
            if response.status_code == 404:
                raise ExpiredBatchError(f"Unknown or expired batch: {job_id}")
            if response.status_code != 200:
                raise BioParserError(
                    f"Batch status failed with HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            last_status = self._json(response, "Batch status")
            status = normalize_text(last_status.get("status")).lower()
            if status in FAILED_JOB_STATUSES:
                raise BioParserError(
                    f"Batch {job_id} failed: "
                    f"{normalize_text(last_status.get('error')) or status}"
                )
            processed = _nonnegative_int(last_status.get("processed"))
            total = _nonnegative_int(last_status.get("total"))
            finished = bool(last_status.get("finished_at"))
            ready = status in SUCCESS_JOB_STATUSES or (
                finished and total > 0 and processed >= total
            )
            if ready:
                result_response = self._request("GET", result_url)
                if result_response.status_code == 404:
                    raise ExpiredBatchError(f"Unknown or expired batch: {job_id}")
                if result_response.status_code == 409:
                    time.sleep(self.poll_interval)
                    continue
                if result_response.status_code != 200:
                    raise BioParserError(
                        "Batch result failed with HTTP "
                        f"{result_response.status_code}: {result_response.text[:1000]}"
                    )
                payload = self._json(result_response, "Batch result")
                outcomes: dict[str, dict[str, Any]] = {}
                for outcome in _list(payload.get("results")):
                    outcome_dict = _dict(outcome)
                    record_id = normalize_text(outcome_dict.get("record_id"))
                    if record_id:
                        outcomes[record_id] = outcome_dict
                return BatchExecutionResult(
                    outcomes=outcomes,
                    cache_hits=_nonnegative_int(last_status.get("cache_hits")),
                    cache_misses=_nonnegative_int(last_status.get("cache_misses")),
                    cache_shared=_nonnegative_int(last_status.get("cache_shared")),
                )
            time.sleep(self.poll_interval)

        raise BioParserError(
            f"Batch {job_id} exceeded job timeout of {self.job_timeout:g} seconds"
        )

    def run_batch(self, items: Sequence[dict[str, str]]) -> BatchExecutionResult:
        last_expired: ExpiredBatchError | None = None
        for attempt in range(self.max_retries + 1):
            submission = self._submit(items)
            try:
                return self._collect_submitted_batch(submission)
            except ExpiredBatchError as exc:
                last_expired = exc
                if attempt >= self.max_retries:
                    break
        raise BioParserError(
            f"Batch expired {self.max_retries + 1} times: {last_expired}"
        )


def resolve_items_with_bisection(
    items: Sequence[dict[str, str]],
    execute_batch: Callable[[Sequence[dict[str, str]]], BatchExecutionResult],
) -> BatchExecutionResult:
    """Split HTTP-422 batches until the rejected item is isolated."""

    if not items:
        return BatchExecutionResult()
    try:
        return execute_batch(items)
    except BatchValidationError as exc:
        if len(items) == 1:
            record_id = items[0]["record_id"]
            return BatchExecutionResult(
                outcomes={
                    record_id: synthetic_failed_outcome(
                        record_id, "batch_validation_error", exc.detail
                    )
                }
            )
        midpoint = len(items) // 2
        combined = BatchExecutionResult()
        combined.merge(resolve_items_with_bisection(items[:midpoint], execute_batch))
        combined.merge(resolve_items_with_bisection(items[midpoint:], execute_batch))
        return combined


def _input_rows(
    dataframe: pd.DataFrame,
    *,
    start_index: int,
    smiles_column: str,
    id_column: str | None,
    label_column: str | None,
) -> list[InputRow]:
    rows: list[InputRow] = []
    for offset, (_, row) in enumerate(dataframe.iterrows()):
        source_index = start_index + offset
        smiles = normalize_text(row.get(smiles_column))
        source_id = normalize_text(row.get(id_column)) if id_column else ""
        if not source_id:
            source_id = f"row_{source_index:09d}"
        expected = normalize_text(row.get(label_column)) if label_column else ""
        rows.append(
            InputRow(
                source_index=source_index,
                source_id=source_id,
                smiles=smiles,
                expected_cyclization=expected,
                api_record_id=api_record_id(smiles) if smiles else "",
            )
        )
    return rows


def _outcome_api_fields(outcome: Mapping[str, Any]) -> dict[str, str]:
    record = _dict(outcome.get("record"))
    error = _dict(outcome.get("error"))
    return {
        "api_processing_status": normalize_text(record.get("processing_status")),
        "api_error_code": normalize_text(error.get("code")),
        "api_error_message": normalize_text(error.get("message")),
    }


def _output_row(
    input_row: InputRow,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    classification = classify_component_graph(outcome)
    return {
        "source_index": input_row.source_index,
        "source_id": input_row.source_id,
        "input_smiles": input_row.smiles,
        "expected_cyclization": input_row.expected_cyclization,
        "api_record_id": input_row.api_record_id,
        "predicted_cyclization": classification.predicted_cyclization,
        "prediction_status": classification.prediction_status,
        "candidate_labels_json": json.dumps(
            classification.candidate_labels, ensure_ascii=True, separators=(",", ":")
        ),
        "component_cycle_rank": classification.component_cycle_rank,
        "closure_count": classification.closure_count,
        "is_multicyclic": classification.is_multicyclic,
        "evidence_json": json.dumps(
            classification.evidence, ensure_ascii=True, separators=(",", ":")
        ),
        **_outcome_api_fields(outcome),
        "prediction_error_code": classification.error_code,
        "prediction_error_message": classification.error_message,
    }


def process_chunk(
    dataframe: pd.DataFrame,
    *,
    start_index: int,
    smiles_column: str,
    id_column: str | None,
    label_column: str | None,
    prefer: str,
    execute_batch: Callable[[Sequence[dict[str, str]]], BatchExecutionResult],
) -> tuple[list[dict[str, Any]], BatchExecutionResult]:
    input_rows = _input_rows(
        dataframe,
        start_index=start_index,
        smiles_column=smiles_column,
        id_column=id_column,
        label_column=label_column,
    )
    unique_items: dict[str, dict[str, str]] = {}
    for row in input_rows:
        if not row.smiles:
            continue
        item = {
            "smiles": row.smiles,
            "record_id": row.api_record_id,
            "prefer": prefer,
        }
        previous = unique_items.get(row.api_record_id)
        if previous is not None and previous["smiles"] != row.smiles:
            raise RuntimeError(f"SHA-256 collision for {row.api_record_id}")
        unique_items[row.api_record_id] = item

    batch_result = resolve_items_with_bisection(
        list(unique_items.values()), execute_batch
    )
    output_rows: list[dict[str, Any]] = []
    for row in input_rows:
        if not row.smiles:
            outcome = synthetic_failed_outcome(
                "", "empty_smiles", "SMILES is empty after normalization"
            )
        else:
            outcome = batch_result.outcomes.get(row.api_record_id)
            if outcome is None:
                outcome = synthetic_failed_outcome(
                    row.api_record_id,
                    "missing_batch_result",
                    "Batch result did not contain the requested record_id",
                )
        output_rows.append(_output_row(row, outcome))
    return output_rows, batch_result


def write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def part_matches_source_indices(path: Path, source_indices: Sequence[int]) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OUTPUT_COLUMNS:
                return False
            observed = [int(row["source_index"]) for row in reader]
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return observed == list(source_indices)


def merge_parts(parts_dir: Path, part_count: int, output_path: Path) -> None:
    temporary = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for part_index in range(part_count):
            part_path = parts_dir / f"part_{part_index:07d}.csv"
            if not part_path.exists():
                raise FileNotFoundError(f"Missing completed part: {part_path}")
            with part_path.open("r", encoding="utf-8", newline="") as part_handle:
                reader = csv.DictReader(part_handle)
                if reader.fieldnames != OUTPUT_COLUMNS:
                    raise ValueError(f"Unexpected columns in part: {part_path}")
                for row in reader:
                    writer.writerow(row)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, output_path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path, max_rows: int | None) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        count = sum(1 for _ in reader)
    return min(count, max_rows) if max_rows is not None else count


def summarize_predictions(
    predictions_path: Path,
    *,
    processing_seconds: float,
    cache_stats: Mapping[str, int],
) -> dict[str, Any]:
    prediction_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = 0
    labeled_rows = 0
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            predicted = row["predicted_cyclization"] or "missing"
            prediction_counts[predicted] += 1
            status_counts[row["prediction_status"] or "missing"] += 1
            error = row["prediction_error_code"] or row["api_error_code"]
            if error:
                error_counts[error] += 1
            expected = row["expected_cyclization"]
            if expected:
                labeled_rows += 1
                confusion[expected][predicted] += 1

    metrics: dict[str, dict[str, float | int | None]] = {}
    if labeled_rows:
        labels = sorted(
            set(confusion)
            | {prediction for row in confusion.values() for prediction in row}
        )
        for label in labels:
            true_positive = confusion[label][label]
            false_positive = sum(
                row[label] for truth, row in confusion.items() if truth != label
            )
            false_negative = sum(
                count for prediction, count in confusion[label].items() if prediction != label
            )
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            metrics[label] = {
                "support": sum(confusion[label].values()),
                "precision": (
                    true_positive / precision_denominator
                    if precision_denominator
                    else None
                ),
                "recall": (
                    true_positive / recall_denominator if recall_denominator else None
                ),
            }

    rows_per_second = total_rows / processing_seconds if processing_seconds > 0 else None
    return {
        "total_rows": total_rows,
        "labeled_rows": labeled_rows,
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "confusion_matrix": {
            truth: dict(sorted(row.items())) for truth, row in sorted(confusion.items())
        },
        "per_class_metrics": metrics,
        "processing_seconds": processing_seconds,
        "rows_per_second": rows_per_second,
        "estimated_remaining_seconds": 0.0,
        "cache_stats": dict(cache_stats),
    }


def write_run_record(
    path: Path,
    *,
    args: argparse.Namespace,
    state: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# Bio Parser Component-Graph Cyclization Classification",
        "",
        "## Runtime",
        f"- Started: {state.get('started_at', '')}",
        f"- Finished: {state.get('finished_at', '')}",
        f"- Host: {socket.gethostname()}",
        f"- Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', sys.prefix)}",
        f"- Python executable: {sys.executable}",
        f"- Command: `{shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])}`",
        "",
        "## Inputs and Outputs",
        f"- Input CSV: {args.input_csv}",
        f"- Input SHA-256: {state.get('input_sha256', '')}",
        f"- Output CSV: {args.output_dir / 'predictions.csv'}",
        f"- SMILES column: {args.smiles_column}",
        f"- ID column: {args.id_column or 'generated from row index'}",
        f"- Label column: {args.label_column or 'none'}",
        "",
        "## Bio Parser",
        f"- Base URL: {args.api_base_url}",
        f"- Prefer mode: {args.prefer}",
        f"- Parser version: {state.get('parser_version', '')}",
        f"- Resource set ID: {state.get('resource_set_id', '')}",
        f"- Batch size: {args.batch_size}",
        "",
        "## Result",
        f"- Total rows: {summary.get('total_rows', 0)}",
        f"- Processing seconds: {summary.get('processing_seconds', 0)}",
        f"- Rows per second: {summary.get('rows_per_second')}",
        "",
        "## Prediction Counts",
    ]
    prediction_counts = _dict(summary.get("prediction_counts"))
    lines.extend(
        f"- {label}: {count}" for label, count in sorted(prediction_counts.items())
    )
    lines.extend(["", "## Error Counts"])
    error_counts = _dict(summary.get("error_counts"))
    if error_counts:
        lines.extend(
            f"- {error}: {count}" for error, count in sorted(error_counts.items())
        )
    else:
        lines.append("- none")
    if summary.get("labeled_rows"):
        lines.extend(["", "## Per-Class Metrics", "", "| Class | Support | Precision | Recall |", "|---|---:|---:|---:|"])
        for label, values in sorted(_dict(summary.get("per_class_metrics")).items()):
            metric = _dict(values)
            precision = metric.get("precision")
            recall = metric.get("recall")
            lines.append(
                f"| {label} | {metric.get('support', 0)} | "
                f"{precision if precision is not None else 'n/a'} | "
                f"{recall if recall is not None else 'n/a'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify cyclization from Bio Parser component graphs without "
            "modifying the source CSV."
        )
    )
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smiles-column", default=DEFAULT_SMILES_COLUMN)
    parser.add_argument("--id-column")
    parser.add_argument("--label-column")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--prefer", choices=("general", "ccd", "helm", "norine"), default="general")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--job-timeout", type=float, default=DEFAULT_JOB_TIMEOUT)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--preflight-only", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.poll_interval <= 0 or args.job_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("Timeouts and poll interval must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.preflight_only:
        return
    if args.input_csv is None or args.output_dir is None:
        raise ValueError("--input-csv and --output-dir are required unless --preflight-only is used")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.input_csv.resolve() == args.output_dir.resolve():
        raise ValueError("Input CSV and output directory must differ")
    header = pd.read_csv(args.input_csv, nrows=0)
    required_columns = [args.smiles_column]
    if args.id_column:
        required_columns.append(args.id_column)
    if args.label_column:
        required_columns.append(args.label_column)
    missing = [column for column in required_columns if column not in header.columns]
    if missing:
        raise KeyError(f"Missing input columns {missing}; available={list(header.columns)}")


def _generated_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "parts",
        output_dir / "predictions.csv",
        output_dir / "summary.json",
        output_dir / "run_record.md",
        output_dir / "run_state.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool, resume: bool) -> None:
    existing = [path for path in _generated_paths(output_dir) if path.exists()]
    if existing and not overwrite and not resume:
        raise FileExistsError(
            "Generated output already exists; pass --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    (output_dir / "parts").mkdir(parents=True, exist_ok=True)


def _state_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_csv": str(args.input_csv.resolve()),
        "smiles_column": args.smiles_column,
        "id_column": args.id_column,
        "label_column": args.label_column,
        "api_base_url": args.api_base_url.rstrip("/"),
        "prefer": args.prefer,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
    }


def _load_or_create_state(
    args: argparse.Namespace,
    *,
    input_hash: str,
    total_rows: int,
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = args.output_dir / "run_state.json"
    configuration = _state_configuration(args)
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume without {state_path}")
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("configuration") != configuration:
            raise ValueError("Resume configuration differs from run_state.json")
        if state.get("input_sha256") != input_hash:
            raise ValueError("Input CSV SHA-256 differs from the resumed run")
        if state.get("parser_version") != capabilities.get("parser_version"):
            raise ValueError("Bio Parser version changed since the resumed run")
        if state.get("resource_set_id") != capabilities.get("resource_set_id"):
            raise ValueError("Bio Parser resource_set_id changed since the resumed run")
        return state
    state = {
        "started_at": iso_now(),
        "finished_at": None,
        "completed": False,
        "configuration": configuration,
        "input_sha256": input_hash,
        "total_rows": total_rows,
        "rows_completed": 0,
        "parts_completed": 0,
        "processing_seconds": 0.0,
        "cache_stats": {"hits": 0, "misses": 0, "shared": 0},
        "parser_version": capabilities.get("parser_version", ""),
        "resource_set_id": capabilities.get("resource_set_id", ""),
        "last_error": None,
    }
    write_json_atomic(state_path, state)
    return state


def _assert_service_identity(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    for field_name in ("parser_version", "resource_set_id"):
        if expected.get(field_name) != observed.get(field_name):
            raise BioParserError(
                f"Service {field_name} changed during the run: "
                f"{expected.get(field_name)!r} -> {observed.get(field_name)!r}"
            )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    with BioParserClient(
        args.api_base_url,
        request_timeout=args.request_timeout,
        job_timeout=args.job_timeout,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries,
    ) as client:
        capabilities = client.preflight(args.prefer)
        if args.preflight_only:
            return capabilities

        prepare_output_dir(
            args.output_dir, overwrite=args.overwrite, resume=args.resume
        )
        input_hash = file_sha256(args.input_csv)
        total_rows = count_csv_rows(args.input_csv, args.max_rows)
        state = _load_or_create_state(
            args,
            input_hash=input_hash,
            total_rows=total_rows,
            capabilities=capabilities,
        )
        state_path = args.output_dir / "run_state.json"
        parts_dir = args.output_dir / "parts"
        source_offset = 0
        part_count = 0
        try:
            reader = pd.read_csv(
                args.input_csv,
                chunksize=args.batch_size,
                nrows=args.max_rows,
            )
            for part_index, dataframe in enumerate(reader):
                part_count = part_index + 1
                source_indices = list(
                    range(source_offset, source_offset + len(dataframe))
                )
                part_path = parts_dir / f"part_{part_index:07d}.csv"
                if args.resume and part_path.exists():
                    if not part_matches_source_indices(part_path, source_indices):
                        raise ValueError(f"Invalid resume part: {part_path}")
                    source_offset += len(dataframe)
                    continue

                part_started = time.monotonic()
                rows, batch_result = process_chunk(
                    dataframe,
                    start_index=source_offset,
                    smiles_column=args.smiles_column,
                    id_column=args.id_column,
                    label_column=args.label_column,
                    prefer=args.prefer,
                    execute_batch=client.run_batch,
                )
                write_csv_atomic(part_path, rows)
                elapsed = time.monotonic() - part_started
                source_offset += len(dataframe)
                state["rows_completed"] = source_offset
                state["parts_completed"] = part_count
                state["processing_seconds"] = float(
                    state.get("processing_seconds", 0.0)
                ) + elapsed
                cache_stats = _dict(state.get("cache_stats"))
                cache_stats["hits"] = _nonnegative_int(cache_stats.get("hits")) + batch_result.cache_hits
                cache_stats["misses"] = _nonnegative_int(cache_stats.get("misses")) + batch_result.cache_misses
                cache_stats["shared"] = _nonnegative_int(cache_stats.get("shared")) + batch_result.cache_shared
                state["cache_stats"] = cache_stats
                state["last_error"] = None
                write_json_atomic(state_path, state)

                processing_seconds = float(state.get("processing_seconds", 0.0))
                rate = source_offset / processing_seconds if processing_seconds > 0 else 0.0
                remaining = max(0, total_rows - source_offset)
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"part={part_index} rows={len(dataframe)} "
                    f"completed={source_offset}/{total_rows} "
                    f"rate={rate:.3f} rows/s eta_seconds={eta:.0f}",
                    flush=True,
                )

            predictions_path = args.output_dir / "predictions.csv"
            merge_parts(parts_dir, part_count, predictions_path)
            final_capabilities = client.preflight(args.prefer)
            _assert_service_identity(capabilities, final_capabilities)
            summary = summarize_predictions(
                predictions_path,
                processing_seconds=float(state.get("processing_seconds", 0.0)),
                cache_stats=_dict(state.get("cache_stats")),
            )
            summary.update(
                {
                    "input_csv": str(args.input_csv.resolve()),
                    "input_sha256": input_hash,
                    "predictions_csv": str(predictions_path),
                    "parser_version": capabilities.get("parser_version", ""),
                    "resource_set_id": capabilities.get("resource_set_id", ""),
                }
            )
            write_json_atomic(args.output_dir / "summary.json", summary)
            state["completed"] = True
            state["finished_at"] = iso_now()
            state["rows_completed"] = total_rows
            state["parts_completed"] = part_count
            write_json_atomic(state_path, state)
            write_run_record(
                args.output_dir / "run_record.md",
                args=args,
                state=state,
                summary=summary,
            )
            return summary
        except Exception as exc:
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["last_updated_at"] = iso_now()
            write_json_atomic(state_path, state)
            raise


def main() -> None:
    args = parse_args()
    try:
        result = run_pipeline(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
