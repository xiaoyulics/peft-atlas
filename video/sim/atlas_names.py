"""Beat 0.2: the method names of the atlas and their count, read from the companion site's data.

site/data/atlas-data.js is one line, `window.ATLAS_DATA = {...};`. The count shown on screen is the number of entries
in ATLAS_DATA.methods (checked against meta.counts.methods and against duplicate names).

  video/.venv/bin/python video/sim/atlas_names.py     # prints the count, writes video/sim/out/atlas_names.json
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = (ROOT / "site" / "data" / "atlas-data.js").read_text(encoding="utf-8")
data = json.loads(src[src.index("{"):src.rstrip().rstrip(";").rindex("}") + 1])
names = [m["name"] for m in data["methods"]]
count = len(names)
meta = data["meta"]["counts"]["methods"]
assert count == meta, f"methods list has {count} entries, meta says {meta}"
assert len(set(names)) == count, "duplicate method names"
print(f"{count} methods in site/data/atlas-data.js (meta.counts.methods = {meta}, built {data['meta']['built']})")
out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
(out / "atlas_names.json").write_text(json.dumps(dict(count=count, built=data["meta"]["built"], names=names),
                                                 ensure_ascii=False, indent=0))
