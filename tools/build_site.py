"""Build index.html (root and docs/) from site/template.html and the data files."""
import json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
d = root / "site/data"
t = (root / "site/template.html").read_text()
mini = lambda p: json.dumps(json.loads((d / p).read_text()), separators=(",", ":"))
html = t.replace("__DATA__", mini("experiments.json")).replace("__CASE01__", mini("case01.json")).replace("__FIGS__", mini("figs.json"))
for p in (root / "index.html", root / "docs/index.html"):
    p.write_text(html)
print("built", len(html))
