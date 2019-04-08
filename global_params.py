# -*- coding: utf-8 -*-
import json
import datetime
import re
from types import SimpleNamespace as sn
cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: sn(**d))

cfg.t = "".join(re.split(r"-|:|\.| ", str(datetime.datetime.now()))[:-1])
