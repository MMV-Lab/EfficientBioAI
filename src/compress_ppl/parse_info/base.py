from abc import ABC, abstractmethod

class BaseParser():
    def __init__(self,config) -> None:
        self.meta_config = config
        
    @abstractmethod
    def parse_model(self):
        pass
    
    @abstractmethod
    def parse_data(self):
        pass
