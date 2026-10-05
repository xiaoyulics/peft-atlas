"""Beat A.1: the atlas as data. Reads site/data/atlas-data.js and writes what the scene draws and counts.

 - counts: methods (382), papers (138), simulation arrows (403 edges among methods) and obstructions (485, summed over
   the methods' obstruction lists), each computed from the data and cross-checked against the data's own bookkeeping;
 - the seven coordinates of every method (theory/axes.json order: kind, shape of the image, base point, gauge,
   covariance group, merge type, rebasing closure) as category indices, rows sorted coordinate by coordinate;
 - the arrows and obstructions as pairs of row indices, and the rows of five named examples.

  video/.venv/bin/python video/sim/atlas.py      # prints the counts, writes video/sim/out/atlas.json
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = (ROOT / "site" / "data" / "atlas-data.js").read_text().strip()
PREFIX = "window.ATLAS_DATA = "
assert src.startswith(PREFIX)
data = json.loads(src[len(PREFIX):].rstrip(";").strip())
axes_file = json.loads((ROOT / "theory" / "axes.json").read_text())["axes"]

AXES = ["kind", "shape", "base_point", "gauge", "covariance", "merge", "rebasing"]
NULL = {"merge": {"na"}}                      # axes.json: 'na' is a legacy key, read as null
methods, papers, edges = data["methods"], data["papers"], data["edges"]
axes = {a["key"]: a for a in data["axes"]}
assert [a["key"] for a in axes_file] == [a["key"] for a in data["axes"]] == AXES

values = {k: [v["key"] for v in axes[k]["values"] if v["key"] not in NULL.get(k, set())] for k in AXES}


def code(m, k):
    v = m["coords"].get(k)
    return -1 if v is None or v in NULL.get(k, set()) else values[k].index(v)


def sort_key(m):
    return tuple(code(m, k) if code(m, k) >= 0 else 99 for k in AXES) + (m["name"].lower(),)


rows = sorted(methods, key=sort_key)
row_of = {m["id"]: i for i, m in enumerate(rows)}
arrows = [(row_of[e["source"]], row_of[e["target"]], e["rule"]) for e in edges]
obstructions = [(row_of[m["id"]], row_of[o["target"]], o["test"]) for m in methods for o in m["obstructions"]]

counts = dict(methods=len(methods), papers=len(papers), arrows=len(edges), obstructions=len(obstructions))
assert counts["methods"] == data["meta"]["counts"]["methods"] and counts["papers"] == data["meta"]["counts"]["papers"]
assert counts["arrows"] == sum(len(m["arrows"]) for m in methods)
print("counts from site/data/atlas-data.js:", counts)
print(f"  arrows: {sum(a == b for a, b, _ in arrows)} self-loops (a method simulating a variant of itself)")
EXAMPLES = ["lora", "oft", "ia3", "dora", "galore"]
for i in EXAMPLES:
    m = rows[row_of[i]]
    print(f"  {m['name']:8s} row {row_of[i]:3d}:", {k: m["coords"].get(k) for k in AXES})
for k in AXES:
    nulls = sum(code(m, k) < 0 for m in methods)
    print(f"  {k:11s} {len(values[k])} values {values[k]}, unset or not applicable: {nulls}")

out = pathlib.Path(__file__).with_name("out")
out.mkdir(exist_ok=True)
(out / "atlas.json").write_text(json.dumps(dict(
    counts=counts, axes=AXES, values=values,
    value_names={k: {v["key"]: v["name"] for v in axes[k]["values"]} for k in AXES},
    rows=[dict(id=m["id"], name=m["name"], codes=[code(m, k) for k in AXES]) for m in rows],
    arrows=arrows, obstructions=obstructions, examples={i: row_of[i] for i in EXAMPLES}), ensure_ascii=False))
