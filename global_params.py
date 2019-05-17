# -*- coding: utf-8 -*-
import sys
import json
from json import JSONEncoder
import datetime
import re
from types import SimpleNamespace

class SuperSimpleNamespace(SimpleNamespace):
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise KeyError("'" + attr + "' is not defined in the configuration. Pass it as a parameter when creating an ECGBatch instance or add it to config.json")

class SimpleNamespaceEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__    

def dump_cfg(fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(cfg, cls=SimpleNamespaceEncoder))

def load_cfg():
    
    t = getattr(sys.modules[__name__], "cfg").t
    cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: SuperSimpleNamespace(**d))
    cfg.t = t
    return cfg

cfg = json.loads(open("config.json", 'r').read(), object_hook=lambda d: SuperSimpleNamespace(**d))
cfg.t = "".join(re.split(r"-|:|\.| ", str(datetime.datetime.now()))[:-1])
