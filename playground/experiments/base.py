from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def run(self):
        ...

    def print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def print_result(self, label: str, value):
        print(f"  {label:<35} {value}")
