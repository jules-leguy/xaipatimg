import ijson
import json
import os
import tempfile

class JSONDB:
    """
    Lazy JSON-backed key–value store.

    Features:
      – Provides dict-like access to a large JSON object without loading it
        fully into memory.
      – Reads are lazy: values are parsed on demand using ijson.
      – Writes are buffered in memory and merged into the JSON file when
        flush() is called.
      – The underlying JSON file must contain a single top-level object
        mapping keys to values.
    """

    def __init__(self, path):
        self.path = path
        self.buffer = {}
        self._load_keys()

        if not os.path.exists(self.path):
            with open(self.path, 'w') as f:
                json.dump({}, f)

    def _load_keys(self):
        if not os.path.exists(self.path):
            self.stored_keys = set()
            return
        keys = []
        with open(self.path, 'rb') as f:
            for prefix, event, value in ijson.parse(f):
                if prefix == '' and event == 'map_key':
                    keys.append(value)
        self.stored_keys = set(keys)

    def __len__(self):
        new_keys_in_buffer = 0
        for k in self.buffer.keys():
            if k not in self.stored_keys:
                new_keys_in_buffer += 1

        return len(self.stored_keys) + new_keys_in_buffer

    def __getitem__(self, key):
        if key in self.buffer:
            return self.buffer[key]
        if key not in self.stored_keys:
            raise KeyError(key)
        with open(self.path, 'rb') as f:
            for k, v in ijson.kvitems(f, ''):
                if k == key:
                    return v
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def flush(self):
        if not self.buffer:
            return

        tmp_path = tempfile.mktemp()

        with open(self.path, 'r') as inp, open(tmp_path, 'w') as out:
            out.write('{\n')
            first = True

            for k, v in ijson.kvitems(inp, ''):
                if k in self.buffer:
                    v = self.buffer.pop(k)
                if not first:
                    out.write(',\n')
                first = False
                out.write(json.dumps({k: v})[1:-1])

            for k, v in self.buffer.items():
                if not first:
                    out.write(',\n')
                first = False
                out.write(json.dumps({k: v})[1:-1])

            out.write('\n}')

        os.replace(tmp_path, self.path)
        self.buffer.clear()
        self._load_keys()

    def items(self):
        """
        Iterate over all key–value pairs in the database in streaming mode.
        """
        self.flush()
        with open(self.path, 'rb') as f:
            for k, v in ijson.kvitems(f, ''):
                yield k, v