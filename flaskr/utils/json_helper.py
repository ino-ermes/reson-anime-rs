from json import JSONEncoder
import json
from flask.json.provider import JSONProvider


class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        return super().default(o)


class CustomJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, **kwargs, cls=CustomJSONEncoder)

    def loads(self, s: str | bytes, **kwargs):
        return json.loads(s, **kwargs)
