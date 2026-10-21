# Paper

Supporting material for *Mechanism-constrained deep learning recovers
individual pharmacokinetics under undernutrition and water-toxicant
co-exposure: an in silico study* (Manne, 2026). The manuscript itself is not
included in this repository.

| File | What it is |
|---|---|
| `MalnuPK_supporting_information.pdf` | Supporting Information, 6 pp, S1–S9 and Figures S1–S2 |
| `supporting_information.md` | Editable source of the Supporting Information |
| `01_population.png` … `06_importance.png`, `02_architecture.png` | Figures |
| `graphical_abstract.png`, `graphical_abstract_1600.png`, `toc.svg` | Graphical abstract and its source |
| `build_pdf.py` | Renders a markdown file to a typeset PDF |

## Rebuilding the Supporting Information PDF

Requires `pandoc` and `playwright` (`pip install playwright && playwright install chromium`).

```bash
pandoc supporting_information.md -t html -s -o s.html
python3 build_pdf.py $PWD/s.html $PWD/MalnuPK_supporting_information.pdf
rm -f s.html
```

The figures are the ones in `../outputs/figures/`, copied here so the
Supporting Information builds standalone. Regenerate them with
`python scripts/make_plots.py --run runs/main_hybrid` from the repository root.
