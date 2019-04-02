# -*- coding: utf-8 -*-
import json
from types import SimpleNamespace as sn
cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: sn(**d))
