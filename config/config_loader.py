import yaml


class ConfigLoader:
    def __init__(self, path):
        self.path = path

    def load(self, extension="yaml"):
        if extension == "yaml":
            return self.load_yaml()
        else:
            raise NotImplementedError()

    def load_yaml(self):
        artifacts = yaml.load(open(self.path, "r"), Loader=yaml.Loader)
        return artifacts
