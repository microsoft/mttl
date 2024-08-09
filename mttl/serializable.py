from dataclasses import asdict, fields


class Serializable:
    @classmethod
    def from_dict(cls, data):
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load_json(cls, path):
        import json

        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
