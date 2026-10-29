from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.analysis.bio_parser_component_graph_batch import (
    BatchExecutionResult,
    BatchValidationError,
    OUTPUT_COLUMNS,
    classify_component_graph,
    part_matches_source_indices,
    process_chunk,
    resolve_items_with_bisection,
    write_csv_atomic,
)


def _outcome(
    *,
    cycle_rank: int,
    unit_ids: list[str],
    edges: list[dict] | None = None,
    connections: list[dict] | None = None,
    ports: list[dict] | None = None,
    connector_groups: list[dict] | None = None,
) -> dict:
    return {
        "contract_id": "parse-outcome",
        "contract_version": "1.0.0",
        "record_id": "fixture",
        "record": {
            "processing_status": "success",
            "covalent_connections": connections or [],
            "attachment_ports": ports or [],
            "polymer_topologies": [{"topology_kind": "contradictory_fixture"}],
            "structural_feature_assertions": [
                {
                    "feature_ref": {
                        "identifier": "covalent-unit-multigraph",
                    },
                    "value": {
                        "schema_version": "1.0.0",
                        "scope": "physical_inter_unit_multigraph",
                        "components": [
                            {"unit_ids": unit_ids, "cycle_rank": cycle_rank}
                        ],
                        "physical_edges": edges or [],
                        "connector_groups": connector_groups or [],
                    },
                }
            ],
        },
        "error": None,
    }


def _connection(
    connection_id: str,
    *,
    role: str,
    chemistry: str | None,
    connection_type: str | None,
    unit_1: str,
    unit_2: str,
    port_1: str,
    port_2: str,
) -> dict:
    return {
        "connection_id": connection_id,
        "role": role,
        "connection_chemistry": chemistry,
        "connection_type": connection_type,
        "endpoint_1": {"unit_id": unit_1, "port_id": port_1},
        "endpoint_2": {"unit_id": unit_2, "port_id": port_2},
    }


def _port(port_id: str, unit_id: str, role: str) -> dict:
    return {"port_id": port_id, "unit_id": unit_id, "role": role}


def _head_to_tail_outcome() -> dict:
    units = ["u_tail", "u_head"]
    return _outcome(
        cycle_rank=1,
        unit_ids=units,
        edges=[
            {
                "physical_edge_id": "bond:head",
                "unit_ids": units,
                "connection_ids": ["connection:head"],
                "roles": ["cyclization"],
            }
        ],
        connections=[
            _connection(
                "connection:head",
                role="cyclization",
                chemistry="amide",
                connection_type="peptide_bond",
                unit_1="u_tail",
                unit_2="u_head",
                port_1="port:tail",
                port_2="port:head",
            )
        ],
        ports=[
            _port("port:tail", "u_tail", "polymerization_acyl"),
            _port("port:head", "u_head", "polymerization_nucleophile"),
        ],
    )


def _disulfide_outcome(*, count: int = 1) -> dict:
    units = [f"u{index}" for index in range(max(2, count + 1))]
    edges = []
    connections = []
    ports = []
    for index in range(count):
        connection_id = f"connection:ss:{index}"
        edge_id = f"bond:ss:{index}"
        unit_1 = units[index]
        unit_2 = units[index + 1]
        port_1 = f"port:ss:{index}:1"
        port_2 = f"port:ss:{index}:2"
        edges.append(
            {
                "physical_edge_id": edge_id,
                "unit_ids": [unit_1, unit_2],
                "connection_ids": [connection_id],
                "roles": ["crosslink"],
            }
        )
        connections.append(
            _connection(
                connection_id,
                role="crosslink",
                chemistry="disulfide",
                connection_type="disulfide",
                unit_1=unit_1,
                unit_2=unit_2,
                port_1=port_1,
                port_2=port_2,
            )
        )
        ports.extend(
            [
                _port(port_1, unit_1, "crosslink"),
                _port(port_2, unit_2, "crosslink"),
            ]
        )
    return _outcome(
        cycle_rank=count,
        unit_ids=units,
        edges=edges,
        connections=connections,
        ports=ports,
    )


def _sc_to_tail_outcome(*, cycle_rank: int = 1) -> dict:
    units = ["u_side", "u_connector", "u_tail"]
    edges = [
        {
            "physical_edge_id": "bond:side",
            "unit_ids": ["u_side", "u_connector"],
            "connection_ids": ["connection:side"],
            "roles": ["crosslink"],
        },
        {
            "physical_edge_id": "bond:tail",
            "unit_ids": ["u_connector", "u_tail"],
            "connection_ids": ["connection:tail"],
            "roles": ["crosslink"],
        },
    ]
    connections = [
        _connection(
            "connection:side",
            role="crosslink",
            chemistry=None,
            connection_type="connector_attachment",
            unit_1="u_side",
            unit_2="u_connector",
            port_1="port:side",
            port_2="port:connector:1",
        ),
        _connection(
            "connection:tail",
            role="crosslink",
            chemistry="amide",
            connection_type="connector_attachment",
            unit_1="u_connector",
            unit_2="u_tail",
            port_1="port:connector:2",
            port_2="port:tail",
        ),
    ]
    ports = [
        _port("port:side", "u_side", "crosslink"),
        _port("port:connector:1", "u_connector", "crosslink"),
        _port("port:connector:2", "u_connector", "crosslink"),
        _port("port:tail", "u_tail", "polymerization_acyl"),
    ]
    groups = [
        {
            "boundary_physical_edge_ids": ["bond:side", "bond:tail"],
            "connector_unit_ids": ["u_connector"],
            "internal_cycle_rank": 0,
            "joins_polymer_ports": True,
            "polymer_endpoints": [
                {"physical_edge_id": "bond:side", "unit_id": "u_side"},
                {"physical_edge_id": "bond:tail", "unit_id": "u_tail"},
            ],
        }
    ]
    return _outcome(
        cycle_rank=cycle_rank,
        unit_ids=units,
        edges=edges,
        connections=connections,
        ports=ports,
        connector_groups=groups,
    )


def check_target_classifications():
    for outcome_factory, expected in (
        (_head_to_tail_outcome, "head_to_tail"),
        (_sc_to_tail_outcome, "sc_to_tail"),
        (_disulfide_outcome, "disulfide_bridge"),
    ):
        result = classify_component_graph(outcome_factory())

        assert result.predicted_cyclization == expected
        assert result.prediction_status == "success"
        assert result.component_cycle_rank == 1
        assert result.closure_count == 1


def test_linear_uses_component_graph_not_polymer_topology():
    result = classify_component_graph(_outcome(cycle_rank=0, unit_ids=["u1", "u2"]))

    assert result.predicted_cyclization == "linear"
    assert result.prediction_status == "success"


def test_unrecognized_cycle_is_other():
    outcome = _outcome(
        cycle_rank=1,
        unit_ids=["u1", "u2"],
        edges=[
            {
                "physical_edge_id": "bond:other",
                "unit_ids": ["u1", "u2"],
                "connection_ids": ["connection:other"],
                "roles": ["crosslink"],
            }
        ],
        connections=[
            _connection(
                "connection:other",
                role="crosslink",
                chemistry="carbon_carbon",
                connection_type="other",
                unit_1="u1",
                unit_2="u2",
                port_1="p1",
                port_2="p2",
            )
        ],
        ports=[_port("p1", "u1", "crosslink"), _port("p2", "u2", "crosslink")],
    )

    result = classify_component_graph(outcome)

    assert result.predicted_cyclization == "other"


def test_conflicting_cycle_types_are_ambiguous():
    head = _head_to_tail_outcome()
    disulfide = _disulfide_outcome()
    record = head["record"]
    graph = record["structural_feature_assertions"][0]["value"]
    ss_record = disulfide["record"]
    ss_graph = ss_record["structural_feature_assertions"][0]["value"]
    graph["components"][0]["cycle_rank"] = 2
    graph["components"][0]["unit_ids"].extend(ss_graph["components"][0]["unit_ids"])
    graph["physical_edges"].extend(ss_graph["physical_edges"])
    record["covalent_connections"].extend(ss_record["covalent_connections"])
    record["attachment_ports"].extend(ss_record["attachment_ports"])

    result = classify_component_graph(head)

    assert result.predicted_cyclization == "ambiguous"
    assert result.prediction_status == "ambiguous"
    assert result.candidate_labels == ["head_to_tail", "disulfide_bridge"]


def test_multiple_same_type_closures_keep_type():
    result = classify_component_graph(_disulfide_outcome(count=2))

    assert result.predicted_cyclization == "disulfide_bridge"
    assert result.closure_count == 2
    assert result.is_multicyclic is True


def test_connector_group_without_cycle_is_linear():
    result = classify_component_graph(_sc_to_tail_outcome(cycle_rank=0))

    assert result.predicted_cyclization == "linear"


def test_missing_component_graph_is_failed():
    result = classify_component_graph(
        {"record": {"processing_status": "success", "structural_feature_assertions": []}}
    )

    assert result.predicted_cyclization == "failed"
    assert result.error_code == "missing_component_graph"


def test_parse_error_is_failed():
    result = classify_component_graph(
        {
            "record": None,
            "error": {"code": "invalid_smiles", "message": "bad input"},
        }
    )

    assert result.predicted_cyclization == "failed"
    assert result.error_code == "invalid_smiles"


def test_422_bisection_isolates_bad_item():
    def execute(items):
        if any(item["smiles"] == "bad" for item in items):
            raise BatchValidationError("bad item")
        return BatchExecutionResult(
            outcomes={
                item["record_id"]: _outcome(cycle_rank=0, unit_ids=["u"])
                for item in items
            }
        )

    items = [
        {"record_id": "ok-1", "smiles": "ok-1", "prefer": "general"},
        {"record_id": "bad", "smiles": "bad", "prefer": "general"},
        {"record_id": "ok-2", "smiles": "ok-2", "prefer": "general"},
    ]
    result = resolve_items_with_bisection(items, execute)

    assert set(result.outcomes) == {"ok-1", "bad", "ok-2"}
    assert result.outcomes["bad"]["error"]["code"] == "batch_validation_error"


def test_process_chunk_preserves_order_and_continues_failed_rows():
    dataframe = pd.DataFrame(
        {
            "sample_id": ["head", "empty", "linear"],
            "SMILES": ["head-smiles", "", "linear-smiles"],
            "Cyclization": ["head_to_tail", "invalid", "linear"],
        }
    )

    def execute(items):
        outcomes = {}
        for item in items:
            outcome = (
                _head_to_tail_outcome()
                if item["smiles"] == "head-smiles"
                else _outcome(cycle_rank=0, unit_ids=["u"])
            )
            outcome["record_id"] = item["record_id"]
            outcomes[item["record_id"]] = outcome
        return BatchExecutionResult(outcomes=outcomes)

    rows, _ = process_chunk(
        dataframe,
        start_index=10,
        smiles_column="SMILES",
        id_column="sample_id",
        label_column="Cyclization",
        prefer="general",
        execute_batch=execute,
    )

    assert [row["source_index"] for row in rows] == [10, 11, 12]
    assert [row["source_id"] for row in rows] == ["head", "empty", "linear"]
    assert [row["predicted_cyclization"] for row in rows] == [
        "head_to_tail",
        "failed",
        "linear",
    ]


def test_atomic_part_validation():
    with tempfile.TemporaryDirectory() as temporary_dir:
        rows = []
        for source_index in (3, 4):
            row = {column: "" for column in OUTPUT_COLUMNS}
            row["source_index"] = source_index
            rows.append(row)
        path = Path(temporary_dir) / "part.csv"

        write_csv_atomic(path, rows)

        assert part_matches_source_indices(path, [3, 4])
        assert not part_matches_source_indices(path, [4, 3])
        with path.open("r", encoding="utf-8", newline="") as handle:
            assert next(csv.reader(handle)) == OUTPUT_COLUMNS


class BioParserComponentGraphTests(unittest.TestCase):
    def test_target_classifications(self):
        check_target_classifications()

    def test_linear_uses_component_graph_not_polymer_topology(self):
        test_linear_uses_component_graph_not_polymer_topology()

    def test_unrecognized_cycle_is_other(self):
        test_unrecognized_cycle_is_other()

    def test_conflicting_cycle_types_are_ambiguous(self):
        test_conflicting_cycle_types_are_ambiguous()

    def test_multiple_same_type_closures_keep_type(self):
        test_multiple_same_type_closures_keep_type()

    def test_connector_group_without_cycle_is_linear(self):
        test_connector_group_without_cycle_is_linear()

    def test_missing_component_graph_is_failed(self):
        test_missing_component_graph_is_failed()

    def test_parse_error_is_failed(self):
        test_parse_error_is_failed()

    def test_422_bisection_isolates_bad_item(self):
        test_422_bisection_isolates_bad_item()

    def test_process_chunk_preserves_order_and_continues_failed_rows(self):
        test_process_chunk_preserves_order_and_continues_failed_rows()

    def test_atomic_part_validation(self):
        test_atomic_part_validation()


if __name__ == "__main__":
    unittest.main()
