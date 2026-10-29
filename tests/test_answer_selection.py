
from collections import Counter

from scripts.evaluate_answers import select_stratified


def make_example(example_id, question_type):
    return {
        "example_id": example_id,
        "question_type": question_type,
        "question": f"Question for {example_id}",
        "answer": "Expected answer",
        "supporting_facts": [["Example", 0]],
    }


def test_selection_uses_question_type():
    examples = [
        make_example("bridge-1", "bridge_comparison"),
        make_example("comparison-1", "comparison"),
        make_example("compositional-1", "compositional"),
        make_example("inference-1", "inference"),
        make_example("bridge-2", "bridge_comparison"),
        make_example("comparison-2", "comparison"),
        make_example("compositional-2", "compositional"),
    ]

    selected = select_stratified(examples, limit=4)

    counts = Counter(
        example["question_type"]
        for example in selected
    )

    assert counts == Counter({
        "bridge_comparison": 1,
        "comparison": 1,
        "compositional": 1,
        "inference": 1,
    })
