"""Tests never connect to a user's database or download neural model weights."""
import os
import sys
import types

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["DEBUG"] = "False"

# Stub out geoalchemy2 so ORM models can be imported without the real package.
# The actual geometry column becomes a plain String under SQLite anyway.
_geo = types.ModuleType("geoalchemy2")
_geo.Geometry = type("Geometry", (), {"__init__": lambda self, *a, **kw: None})
_geo.Geography = type("Geography", (), {"__init__": lambda self, *a, **kw: None})
_geo_funcs = types.ModuleType("geoalchemy2.functions")
_geo_funcs.ST_DWithin = lambda *a, **kw: None
_geo_funcs.ST_MakePoint = lambda *a, **kw: None
_geo_funcs.ST_SetSRID = lambda *a, **kw: None
_geo_elements = types.ModuleType("geoalchemy2.elements")
_geo_elements.WKTElement = type("WKTElement", (), {"__init__": lambda self, *a, **kw: None})
_geo_types = types.ModuleType("geoalchemy2.types")
sys.modules.setdefault("geoalchemy2", _geo)
sys.modules.setdefault("geoalchemy2.functions", _geo_funcs)
sys.modules.setdefault("geoalchemy2.elements", _geo_elements)
sys.modules.setdefault("geoalchemy2.types", _geo_types)

