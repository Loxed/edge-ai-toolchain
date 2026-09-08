import csv
from pathlib import Path


class LogicGatesDataset:
    def __init__(self, path: str | Path):
        self.samples: list[tuple[list[int], int]] = []

        with open(path, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                x = [int(row["x1"]), int(row["x2"])]
                y = int(row["y"])
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.samples[index]