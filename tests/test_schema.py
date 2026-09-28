"""clm.schema is the contract between the server and the trainer: these tests pin the
state / candidate texts the heads see and how distributions become Answers."""
import math

import pytest

from clm.schema import (answer_from_logits, answer_from_probs, build_pairs, candidates, confidence, label_of,
                        probabilities_of, softmax, state_text, to_text)


def test_to_text_renders_objects_and_arrays_as_prose():
    assert to_text(None) == ""
    assert to_text(True) == "true"
    assert to_text(3) == "3"
    assert to_text({"a": 1, "b": "x"}) == "a: 1\n\nb: x"
    assert to_text({"a": {"b": 1, "c": 2}}) == "a:\n  b: 1\n  c: 2"
    assert to_text(["x", "y"]) == "- x\n- y"
    assert to_text([{"k": "v"}]) == "-\n  k: v"


def test_state_text_puts_question_after_context():
    assert state_text("Customer: hi", "Is this urgent?") == "Customer: hi\n\nIs this urgent?"
    assert state_text("  ", "Is this urgent?") == "Is this urgent?"
    assert state_text("Customer: hi", None) == "Customer: hi"


def test_noul_candidates_default_to_the_statement():
    keys, texts = candidates({"type": "noul", "instructions": "Is this urgent?"})
    assert keys == ["false", "true"]
    assert texts == ["false: No. This is false: Is this urgent?", "true: Yes. This is true: Is this urgent?"]


def test_noul_candidates_use_given_descriptions():
    _, texts = candidates({"type": "noul", "instructions": "q", "criteria": {"true": "It is urgent."}})
    assert texts == ["false: No. This is false: q", "true: It is urgent."]


def test_choice_candidates_use_description_else_key():
    keys, texts = candidates({"type": "choice", "criteria": {"billing": "A billing issue", "tech": ""}})
    assert keys == ["billing", "tech"]
    assert texts == ["A billing issue", "tech"]


def test_score_candidates_are_indexed_levels():
    keys, texts = candidates({"type": "score", "criteria": ["Calm", "Frustrated", "Very angry"]})
    assert keys == ["0", "1", "2"]
    assert texts == ["Calm", "Frustrated", "Very angry"]


@pytest.mark.parametrize("q", [
    {"type": "yesno"},
    {"type": "choice", "criteria": {}},
    {"type": "choice", "criteria": ["a", "b"]},
    {"type": "score", "criteria": ["only one"]},
])
def test_invalid_questions_raise(q):
    with pytest.raises(ValueError):
        candidates(q)


def test_build_pairs_per_question():
    pairs = build_pairs("S", {"u": {"type": "noul", "instructions": "Urgent?"},
                              "d": {"type": "choice", "instructions": "Dept?", "criteria": {"a": "", "b": ""}}})
    assert pairs["u"][0] == "S\n\nUrgent?"
    assert pairs["d"] == ("S\n\nDept?", ["a", "b"], ["a", "b"])


def test_softmax_is_stable_and_normalised():
    p = softmax([1000.0, 1000.0, 0.0])
    assert math.isclose(sum(p), 1.0)
    assert math.isclose(p[0], 0.5) and p[2] < 1e-300


def test_confidence_is_top_minus_mean_of_rest():
    assert confidence([1.0]) == 1.0
    assert math.isclose(confidence([0.7, 0.2, 0.1]), 0.7 - 0.15)
    assert confidence([0.5, 0.5]) == 0.0


def test_answer_from_probs_per_type():
    noul = answer_from_probs({"type": "noul"}, ["false", "true"], [0.3, 0.7])
    assert noul == {"type": "noul", "noul": 0.7}

    choice = answer_from_probs({"type": "choice"}, ["a", "b"], [0.2, 0.8])
    assert choice["choice"] == "b" and choice["probabilities"] == {"a": 0.2, "b": 0.8}

    score = answer_from_probs({"type": "score", "criteria": ["lo", "mid", "hi"]}, ["0", "1", "2"], [0.25, 0.5, 0.25])
    assert math.isclose(score["score"], 1.0)
    assert score["legend"] == {"0": "lo", "1": "mid", "2": "hi"}


def test_answer_from_logits_matches_softmax():
    a = answer_from_logits({"type": "choice"}, ["a", "b"], [0.0, math.log(3)])
    assert math.isclose(a["probabilities"]["b"], 0.75)


def test_label_and_probabilities_of():
    assert label_of({"type": "noul", "noul": 0.5}) == "true"
    assert label_of({"type": "noul", "noul": 0.49}) == "false"
    assert label_of({"type": "choice", "choice": "b"}) == "b"
    assert label_of({"type": "score", "probabilities": {"0": 0.1, "1": 0.9}}) == "1"
    assert probabilities_of({"type": "noul", "noul": 0.25}) == {"false": 0.75, "true": 0.25}
