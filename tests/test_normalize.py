
from src.ingestion.normalize import (
    example_to_documents,
    example_to_evaluation_item,
)


def make_example():
    return {
        "_id": "example-1",
        "type": "bridge_comparison",
        "question": "Which director was younger?",
        "answer": "Director B",
        "context": [
            [
                "Director A",
                [
                    "Director A was born in 1940.",
                    "Director A directed Film A.",
                ],
            ],
            [
                "Director B",
                [
                    "Director B was born in 1960.",
                ],
            ],
        ],
        "supporting_facts": [
            ["Director A", 0],
            ["Director B", 0],
        ],
        "evidences": [],
    }


def test_example_becomes_two_passages():
    documents = example_to_documents(make_example())

    assert len(documents) == 2
    assert documents[0]["title"] == "Director A"
    assert documents[0]["sentences"] == [
    "Director A was born in 1940.",
    "Director A directed Film A.",
    ]


def test_supporting_sentence_indices_are_preserved():
    documents = example_to_documents(make_example())

    assert (
        documents[0]["provenance"][0][
            "supporting_sentence_indices"
        ]
        == [0]
    )


def test_identical_passages_have_identical_ids():
    first = make_example()
    second = make_example()

    second["_id"] = "example-2"

    first_document = example_to_documents(first)[0]
    second_document = example_to_documents(second)[0]

    assert (
        first_document["document_id"]
        == second_document["document_id"]
    )


def test_answers_stay_out_of_searchable_documents():
    example = make_example()

    documents = example_to_documents(example)
    evaluation = example_to_evaluation_item(example)

    assert "answer" not in documents[0]
    assert "question" not in documents[0]

    assert evaluation["answer"] == "Director B"
    assert evaluation["question"] == "Which director was younger?"
    assert evaluation["supporting_facts"] == [
        ["Director A", 0],
        ["Director B", 0],
    ]
