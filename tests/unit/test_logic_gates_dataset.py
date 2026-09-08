from edgekit.datasets.logic_gates import LogicGatesDataset


def test_loads_logic_gate_dataset():
    dataset = LogicGatesDataset("data/logic_gates/and.csv")

    assert len(dataset) == 4
    assert dataset[0] == ([0, 0], 0)
    assert dataset[1] == ([0, 1], 0)
    assert dataset[2] == ([1, 0], 0)
    assert dataset[3] == ([1, 1], 1)

    dataset = LogicGatesDataset("data/logic_gates/xor.csv")
    assert len(dataset) == 4
    assert dataset[0] == ([0, 0], 0)
    assert dataset[1] == ([0, 1], 1)
    assert dataset[2] == ([1, 0], 1)
    assert dataset[3] == ([1, 1], 0)