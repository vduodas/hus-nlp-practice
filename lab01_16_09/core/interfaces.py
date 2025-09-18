from abc import abstractmethod, ABC

class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass