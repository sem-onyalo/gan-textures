import os

from storage.storage import Storage

class StorageLocal(Storage):
    def __init__(self, root_directory) -> None:
        super().__init__(root_directory)

    def write_bytes(self, filename, data, path_parts=[]):
        local_directory = self.root_directory

        if len(path_parts) > 0:
            local_directory = os.path.join(local_directory, *path_parts)

        os.makedirs(local_directory, exist_ok=True)

        file_path = os.path.join(local_directory, filename)
        with open(file_path, mode="wb") as fd:
            fd.write(data)
