# Paper

| File | What it is |
|---|---|
| `MalnuPK_manuscript.pdf` | The preprint, 29 pp, figures embedded |
| `MalnuPK_supporting_information.pdf` | Supporting Information, 6 pp, S1–S9 and Figures S1–S2 |
| `manuscript.md`, `supporting_information.md` | Editable sources |
| `graphical_abstract.png`, `toc.svg` | Graphical abstract and its source |
| `SUBMISSION.md` | Preprint submission notes: abstract, keywords, category, licence, checklist |
| `build_pdf.py` | Renders either markdown file to a typeset PDF |

## Rebuilding the PDFs

Requires `pandoc` and `playwright` (`pip install playwright && playwright install chromium`).

```bash
python3 - <<'PY'
import subprocess, pathlib
for src, out in (("manuscript.md","m.html"), ("supporting_information.md","s.html")):
    body = subprocess.run(["pandoc", src, "-t", "html"], capture_output=True, text=True).stdout
    pathlib.Path(out).write_text("<!doctype html><html><head><meta charset='utf-8'>"
                                 "</head><body>\n" + body + "\n</body></html>")
PY
python3 build_pdf.py $PWD/m.html $PWD/MalnuPK_manuscript.pdf
python3 build_pdf.py $PWD/s.html $PWD/MalnuPK_supporting_information.pdf
rm -f m.html s.html
```

The figures in the manuscript are the ones in `../outputs/figures/`, copied here
so the paper builds standalone. Regenerate them with
`python scripts/make_plots.py --run runs/main_hybrid` from the repository root.
