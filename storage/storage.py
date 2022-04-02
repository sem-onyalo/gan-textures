class Storage:
    def __init__(self, root_directory) -> None:
        assert root_directory != ""
        self.root_directory = root_directory

    def write_bytes(self, filename, data, path_parts=[]):
        pass
