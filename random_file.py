from abc import ABC
from abc import abstractmethod


def square(x: float):
    addition = x + 1
    square = addition**2
    return square


a = 34


class King(ABC):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    @abstractmethod
    def get_info(self) -> dict:
        """Gets the info of the king as a dictionary

        Returns: The information of the king


        Raises:
            NotImplementedError:
        """
        raise NotImplementedError("Implement this brother ")

    def get_name(self) -> str:
        """Gets the name of the person

        Returns:
            The name of the king

        """
        return self.name

    def get_age(self) -> int:
        """Gets the age of the person

        Returns:
            The age
        """
        return self.age


class Blaise(King):
    """Some random class

    Attributes:
        name: the name of the person
        age: The age of the person
    """

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def get_info(self):
        return {"name": self.get_name(), "age": self.get_age()}


if __name__ == "__main__":
    some_square = square(23)
    me = Blaise("Blaise", 23)
