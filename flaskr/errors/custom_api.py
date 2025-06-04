class CustomAPIError(Exception):

    def __init__(self, message):
        super().__init__()
        self.message = message

    def to_dict(self):
        rv = dict()
        rv['message'] = self.message
        if hasattr(self, 'status_code'):
            rv['status code'] = self.status_code
        return rv