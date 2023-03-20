from abc import ABC, abstractmethod

class BaseInfer():
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def prepare_data(self):
        pass
    
    @abstractmethod
    def run_infer(self):
        pass 
    
    @abstractmethod
    def save_result(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def calculate_infer_time(self):
        pass
    
    @abstractmethod
    def calculate_energy(self):
        pass