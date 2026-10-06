import json

import pytest

import adapters


def _row(gold, **kw):
    return {"id": "r1", "state": "Customer: my card was charged twice",
            "questions": json.dumps({"dept": {"type": "choice", "instructions": "Which team?",
                                              "criteria": {"billing": "", "tech": ""}}}),
            "gold": json.dumps(gold), **kw}


def test_typed_decision_examples_from_json_encoded_rows():
    (e,) = adapters.typed_decision_examples([_row({"dept": {"label": "tech"}}, workflow="support")])
    assert e.qid == "dept" and e.group == "r1" and e.workflow == "support"
    assert e.state_text == "Customer: my card was charged twice\n\nWhich team?"
    assert e.keys == ["billing", "tech"] and e.candidates == ["billing", "tech"]
    assert e.label == 1 and e.keys[e.label] == "tech"
    assert e.target == [0.0, 1.0]


def test_soft_target_is_normalised_over_keys():
    (e,) = adapters.typed_decision_examples(
        [_row({"dept": {"label": "billing", "probabilities": {"billing": 3, "tech": 1}}})])
    assert e.target == [0.75, 0.25]


def test_bool_noul_labels_map_to_keys():
    row = {"state": "s", "questions": {"u": {"type": "noul", "instructions": "Urgent?"}},
           "gold": {"u": {"label": True}}}
    (e,) = adapters.typed_decision_examples([row])
    assert e.keys == ["false", "true"] and e.label == 1


def test_questions_without_gold_are_skipped():
    assert list(adapters.typed_decision_examples([_row({})])) == []


def test_unknown_gold_label_raises():
    with pytest.raises(ValueError, match="not among options"):
        list(adapters.typed_decision_examples([_row({"dept": {"label": "sales"}})]))


def test_read_transitions_jsonl_and_missing_keys(tmp_path):
    rec = {"state": [], "action": "a", "task_id": "t", "step_idx": 0}
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(rec) + "\n\n")
    assert adapters.read_transitions(str(p)) == [rec]

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"state": []}]))
    with pytest.raises(ValueError, match="missing"):
        adapters.read_transitions(str(bad))
