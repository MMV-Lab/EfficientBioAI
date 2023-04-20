from abc import ABC, abstractmethod  # noqa: F401


class BaseParser:
    def __init__(self, config) -> None:
        self.meta_config = config

    @abstractmethod
    def parse_model(self):
        pass

    @abstractmethod
    def parse_data(self):
        pass

    @abstractmethod
    def fine_tune(self):
        pass

    @abstractmethod
    def calibrate(self):
        pass
