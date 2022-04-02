from typing import List

from storage.local import StorageLocal
from storage.storage import Storage

class StorageManager:
    def __init__(self, root_directory) -> None:
        self.storages:List[Storage] = []
        self.storages.append(StorageLocal(root_directory))

    def write_bytes(self, filename, data, path_parts=[]) -> None:
        for storage in self.storages:
            storage.write_bytes(filename, data, path_parts)