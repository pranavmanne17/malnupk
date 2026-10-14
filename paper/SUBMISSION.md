# ChemRxiv submission pack

Author details and ORCID are filled in. **One placeholder remains**: the
repository URL or DOI, marked `[REPOSITORY URL / DOI]` in `manuscript.md`.

---

## 1. Files to upload

| Slot | File |
|---|---|
| Main manuscript (PDF, required) | `MalnuPK_manuscript.pdf` — 29 pp, figures embedded |
| Supporting information | `MalnuPK_supporting_information.pdf` — 6 pp, S1–S9 + Figures S1–S2 |
| Graphical/TOC abstract | `graphical_abstract.png` — 3200×1520; `graphical_abstract_1600.png` if a smaller file is wanted |
| Supplementary (code) | `malnupk_code.tar.gz` — or, better, deposit on Zenodo/GitHub and paste the DOI/URL |

ChemRxiv accepts the main manuscript as PDF only; supplementary files can be any
type. Editable sources are included (`manuscript.md`, `supporting_information.md`,
`toc.svg`, `build_pdf.py`) so you can regenerate after filling in your details.

## 2. Title

> Mechanism-constrained deep learning recovers individual pharmacokinetics under
> undernutrition and water-toxicant co-exposure: an in silico study

## 3. Abstract (plain text, ~250 words — for the submission form)

> Children with severe acute malnutrition are dosed in milligrams per kilogram,
> a rule that treats every kilogram of body mass as pharmacokinetically
> equivalent. Undernutrition violates that assumption along several axes at
> once, and in opposite directions for different drugs. These children are also
> disproportionately exposed to contaminated drinking water, an exposure
> mechanistically coupled to their nutritional state rather than independent of
> it.
>
> We built a generative simulator in which nutritional status and drinking-water
> toxicant exposure jointly determine body composition, organ mass, enzyme
> capacity, protein binding and gut integrity, and from these derive the
> pharmacokinetic parameters of nine essential medicines. A model was trained on
> 36,000 simulated concentration-time records from 12,000 virtual patients. It
> receives only bedside-observable covariates, predicts seven physiological
> parameters, and a differentiable compartmental solver converts them to a
> concentration-time curve; allometric exponents are fixed and dose is excluded
> from the encoder.
>
> Against noise-free simulated truth the model reached a median fold error of
> 1.050 (98.6% within twofold) and recovered the latent parameters, never
> supplied as inputs, with R-squared 0.981-0.999. In a matched comparison the
> mechanistic solver was worth 1.54-fold in log-RMSE and auxiliary supervision
> 3.72-fold. The model did not transfer to a held-out drug (fold error 11.2) and
> Monte Carlo dropout was substantially overconfident. Comparison against four
> published sources exposed two structural defects in the simulator that 47 unit
> tests had missed.
>
> These results measure recovery of a specified generative process, not human
> pharmacokinetics, and support no clinical inference.

Word count: 238. The standalone synthetic-data statement has been removed from
the manuscript at your request; add your own wherever you want it, and mirror it
here if you want it in the posted abstract.

## 4. Keywords

pharmacokinetics; severe acute malnutrition; physiologically based modelling;
scientific machine learning; dose individualisation; environmental toxicants;
paediatric pharmacology; uncertainty calibration

## 5. Category

Primary: **Biological and Medicinal Chemistry → Pharmacology** (or
**Theoretical and Computational Chemistry** if the form steers that way).

**One caveat on venue.** ChemRxiv's stated scope is "chemistry and related
areas", and screening includes a scope check. This is a pharmacometrics and
machine-learning paper with no chemistry experiment in it, so there is a real
chance of a scope rejection. If that happens, **bioRxiv** (Pharmacology and
Toxicology, or Scientific Communication and Education) or **arXiv** (q-bio.QM)
is the natural home, and the same two PDFs work unchanged. Nothing in the
package is ChemRxiv-specific.

## 6. Licence

**CC BY 4.0** recommended — ChemRxiv offers CC BY, CC BY-NC and CC BY-NC-ND.
CC BY is the least restrictive and the usual default for work you want cited and
reused; it does not stop you publishing in a journal afterwards.

## 7. Declarations to paste into the form

- **Funding:** This research received no specific grant from any funding agency
  in the public, commercial or not-for-profit sectors.
- **Conflict of interest:** The author declares no competing interests.
- **Ethical approval:** Not applicable. No human participants, animals,
  biological samples or identifiable data were involved.
- **Data availability:** All code, trained weights and evaluation outputs are
  openly available at [URL/DOI]. No empirical dataset exists or is required; the
  synthetic dataset regenerates deterministically from a fixed seed.

## 8. One placeholder left

`[REPOSITORY URL / DOI]` in `manuscript.md` — see section 9. Everything else
(author, affiliation, email, ORCID 0009-0004-7575-7208, funding, competing
interests) is filled in.

Then rebuild:

```bash
python3 - <<'EOF'
import subprocess, pathlib
for src, out in (("manuscript.md","m.html"), ("supporting_information.md","s.html")):
    body = subprocess.run(["pandoc", src, "-t", "html"], capture_output=True, text=True).stdout
    pathlib.Path(out).write_text("<!doctype html><html><head><meta charset='utf-8'>"
                                 "</head><body>\n" + body + "\n</body></html>")
EOF
python3 build_pdf.py $PWD/m.html $PWD/MalnuPK_manuscript.pdf
python3 build_pdf.py $PWD/s.html $PWD/MalnuPK_supporting_information.pdf
rm -f m.html s.html
```

## 9. Code deposit

ChemRxiv screens for code availability. A tarball attached as supplementary
material satisfies it, but a citable archive is better and takes ten minutes:
push the repo to GitHub, link the repository to Zenodo, cut a release, and
Zenodo mints a DOI. Paste that DOI into `[REPOSITORY URL / DOI]`.

## 10. Before you click submit

- [ ] Repository link filled in, PDFs rebuilt
- [ ] You have read the manuscript end to end and agree with every claim in it
- [ ] In particular, you are comfortable defending: the solver being worth
      1.54× and auxiliary supervision 3.72× (this reverses the earlier draft);
      the gentamicin toxicity threshold change; and the statement that nine
      claims and four pharmacodynamic targets remain unsourced
- [ ] Licence chosen
- [ ] Code deposited and the link live

Posting takes up to 48 hours after submission. Versions are permanent — later
versions are accepted and earlier ones stay publicly visible with a timestamp,
so a v1 with placeholders left in would be visible forever. Check the four
fields.
