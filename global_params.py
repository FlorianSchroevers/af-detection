# -*- coding: utf-8 -*-
import sys
import json
from json import JSONEncoder
import datetime
import re
from types import SimpleNamespace as sn

class SimpleNamespaceEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__    

def dump_cfg(fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(cfg, cls=SimpleNamespaceEncoder))

def load_cfg():
	
	t = getattr(sys.modules[__name__], "cfg").t
	cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: sn(**d))
	cfg.t = t
	return cfg

cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: sn(**d))
cfg.t = "".join(re.split(r"-|:|\.| ", str(datetime.datetime.now()))[:-1])
