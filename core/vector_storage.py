from abc import ABC, abstractmethod

class IVectorStorage(ABC):

    @abstractmethod
    async def add(p): ...

    @abstractmethod
    async def search(collection ,vector, k, filters): ...

    @abstractmethod
    async def delete(id): ...
