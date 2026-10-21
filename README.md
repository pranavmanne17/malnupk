# MalnuPK

**A pharmacology-informed deep learning model of drug disposition in
undernourished, environmentally-exposed children.**

[![tests](https://github.com/pranavmanne17/malnupk/actions/workflows/tests.yml/badge.svg)](https://github.com/pranavmanne17/malnupk/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pranav Manne ([ORCID 0009-0004-7575-7208](https://orcid.org/0009-0004-7575-7208))


```bash
git clone https://github.com/pranavmanne17/malnupk && cd malnupk
pip install -r requirements.txt
pytest -q tests/                 # 47 tests, ~5 s
bash scripts/run_all.sh          # full pipeline, ~2.5 h on two CPU cores
```

Every number in the paper is in `outputs/`, so the results can be checked
without re-running anything. Trained weights for all nine runs are included in
`runs/*/best.pt` (about 2.7 MB each), and they regenerate exactly from a fixed
seed. See `runs/README.md`.

## Repository layout

```
malnupk/     the package: simulator, drug library, solver, model, dosing, evaluation
scripts/     dataset generation, training, evaluation, figures, diagnostics
tests/       47 tests, weighted toward physical correctness
outputs/     every number in the paper, plus the six figures  (see outputs/README.md)
runs/        trained weights and training histories for all nine runs
paper/       Supporting Information, figures, graphical abstract
REFERENCES.md   30 verified sources, and the claims that still lack one
RESULTS.md      generated from outputs/ by scripts/summarise.py
CHANGELOG.md    how the results changed as the simulator was corrected
```


## The problem

Children with severe acute malnutrition are dosed in milligrams per kilogram.
That rule assumes every kilogram of body mass is pharmacokinetically
interchangeable. In undernutrition it is not, and it fails in at least six
directions at once:

| Mechanism | Consequence |
|---|---|
| Fat is lost faster than lean tissue | water-soluble drugs gain volume per kg, lipophilic drugs lose it |
| Liver and kidney shrink *less* than body weight ("organ sparing") | clearance per kg rises while absolute clearance falls |
| Inflammation and protein deficit suppress CYP3A4; fasting spares CYP2E1 | metabolic routes re-weight, changing both efficacy and toxic-metabolite load |
| Albumin falls and α1-acid glycoprotein rises | unbound fraction moves in **opposite** directions for acidic and basic drugs |
| Environmental enteric dysfunction blunts villi, raises gastric pH, depletes bile salts | oral absorption of lipophilic drugs collapses |
| Cardiac output falls | delivery to eliminating organs drops for high-extraction drugs |

These children are also largely absent from the trials that set the labels, so
the evidence to fix the dose does not exist. And they disproportionately drink
contaminated water, which is not an independent nuisance but a *coupled* second
axis: iron deficiency up-regulates DMT1, which also transports lead and
cadmium, so a malnourished child absorbs several-fold more metal from the same
well; protein deficit starves the S-adenosyl-methionine pool that arsenic
methylation depends on, so the same intake produces a larger internal burden;
and cadmium then reduces GFR, arsenic injures hepatocytes, and lead impairs
haem synthesis — which is the prosthetic group of every cytochrome P450.

MalnuPK builds that entire causal chain explicitly, simulates virtual trials
through it, and trains a neural network to invert it from the handful of
measurements a clinician can actually obtain.

### Relation to existing work

A physiologically-based pharmacokinetic framework for malnourished children
already exists (Sjögren et al., *Pharmaceutics* 2021;13(2):204), and a
systematic review covers what is actually measured in this population
(Verrest et al., *Clin Pharmacokinet* 2021;60(9):1149–1169, 49 studies,
29 drugs). MalnuPK is not the first mechanistic model here. What is different
is the inverse direction — learning to recover individual parameters from
bedside-observable covariates rather than forward-simulating a typical patient
— the explicit toxicant axis, and the dose-decision layer on top. Those
distinctions are the contribution; the mechanistic modelling of malnutrition
itself is not novel and is not claimed to be.

## What this repository contains

```
malnupk/
  physiology.py   virtual population: anthropometry → body composition → organ
                  mass → blood flow → GFR → enzyme capacity → gut function
  toxicants.py    water chemistry → intake → body burden → effects on the above
  drugs.py        nine essential medicines with physicochemical descriptors
  pk.py           physiology → PK parameters → exact linear-system solution
  schedule.py     the dosing/observation schedule shared by simulator and model
  dataset.py      virtual-trial generation, realistic sparse sampling, splits
  features.py     the observable/latent contract (see below)
  torch_pk.py     the same solver, differentiable and batched
  model.py        the network
  dosing.py       dose individualisation and therapeutic drug monitoring
  evaluate.py     accuracy, calibration, attribution, clinical comparison
  train.py        training loop
  cli.py          command-line interface
scripts/          dataset generation, training, evaluation, figures
tests/            47 tests, weighted toward physical correctness
```

---

## Three design decisions that carry the project

### 1. The model sees only what a clinician sees

The simulator knows each virtual child's liver mass, GFR, CYP3A4 activity and
unbound fraction. A clinician in a stabilisation ward does not. Feeding those
latent quantities to the network would make the learning problem trivial and
the result useless.

So the **inputs** are anthropometry, a small bedside biochemistry panel
(*each value independently allowed to be missing*, because often it is),
drinking-water chemistry from a field test kit, two internal exposure
biomarkers, and the drug's own descriptors.

The latent physiology is kept as **auxiliary supervision**. The network is
asked to predict it as a side task, which regularises the shared encoder
toward a physiologically meaningful latent space and makes the model
inspectable — you can ask *why* it expects under-exposure and get back
"inferred CYP3A4 at 32% of reference and bile-salt-dependent absorption
halved", not a SHAP plot.

As it turns out this is the single most important design choice in the
project, and by a wide margin — see the measured contribution below.

### 2. The network predicts parameters; mechanism predicts the curve

The model never regresses concentrations. It predicts seven physiological
parameters — CL, V1, Q, V2, ka, F, fu — and a **differentiable compartmental
solver** turns them into a concentration-time curve by exactly solving the
same linear system the simulator used (agreement verified to 1e-13).

```
log θ  =  h_drug(descriptors)  +  allometry(weight)  +  g_patient(covariates, drug)
          └ places an unseen drug    └ fixed 0.75/1.0    └ the learned departure
                                        exponents           from the mg/kg rule
```

This buys four things that a black-box curve regressor cannot have:

- **Physical admissibility.** Every output is a real solution of a mass-balance
  system with non-negative parameters. No negative concentrations, no curves
  that rise without a dose, no violations of superposition.
- **Exact dose extrapolation.** Dose enters the *solver*, not the weights, and
  is deliberately excluded from the parameter encoder. Doubling the dose
  doubles the curve to ten significant figures, including at doses never seen
  in training. This is what makes inverse dosing sound.
- **Interpretability.** The bottleneck is seven numbers a pharmacologist
  recognises.
- **Data efficiency.** Two or three opportunistic samples per child constrain
  seven parameters; they come nowhere near constraining a free-form curve.

Because the allometric exponents are **hard-wired rather than learned**, every
body-size effect the network does learn is by construction a *departure* from
the allometric rule — which is exactly the quantity of interest, since the
allometric rule is what mg/kg dosing already assumes.

### 3. Covariates are read through the drug (FiLM conditioning)

A covariate has no drug-independent meaning. Albumin dominates ceftriaxone and
is irrelevant to gentamicin. Bile-salt deficiency destroys lumefantrine and
does nothing to amoxicillin. The drug embedding therefore **modulates** the
patient encoder's features multiplicatively rather than being concatenated to
them, so the same measurement is read differently depending on the drug.

---

## Quick start

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn pytest

python scripts/make_dataset.py --n-patients 12000 --drugs-per-patient 3
python scripts/train_all.py    --epochs 30               # all runs + ablations
python scripts/evaluate.py     --all
python scripts/clinical_impact.py --run runs/main_hybrid --n 120
python scripts/make_plots.py   --run runs/main_hybrid
python scripts/summarise.py                              # writes RESULTS.md
pytest -q tests/
```

Or the whole pipeline end to end (~2.5 hours on two cores):

```bash
bash scripts/run_all.sh
```

### Command line

```bash
# What happens to a severely oedematous 20-month-old on label-dose rifampicin?
python -m malnupk.cli simulate --whz -3.4 --age 20 --oedema 2 \
                               --albumin 17 --drug rifampicin

# What dose would the model give instead?
python -m malnupk.cli dose --whz -3.4 --age 20 --oedema 2 \
                           --albumin 17 --drug rifampicin

# Refine from two measured gentamicin levels
python -m malnupk.cli tdm --drug gentamicin --whz -3.0 \
                          --levels 1.0:18.4,12.0:1.1

python -m malnupk.cli population --n 500
python -m malnupk.cli drugs
```

---

## External validation

**This is the part that distinguishes the project from a self-consistent simulation.**
Full detail in [`RESULTS.md`](RESULTS.md) and `outputs/external_validation.json`;
reproduce with `python scripts/validate_external.py`.

Comparing the simulator against four published sources found two structural
defects that the unit-test suite had not caught:

- **Organ sparing was inert.** The sparing exponent was multiplied onto body
  surface area computed from *actual* weight, which already shrinks with the
  lost mass, so the two cancelled exactly. Kidney mass moved from 13.3 to
  13.4 g/kg across a twofold range of wasting — the mechanism was in the code
  and not in the model.
- **Toxicant potencies were on an arbitrary scale.** They implied that water
  meeting every WHO guideline would cost a child ~31% of their glomerular
  filtration rate. Potencies are now expressed as multiples of each
  toxicant's guideline-equivalent burden, which makes them checkable.

Neither was findable from inside the system: every component behaved as
specified, and the error was in how they composed.

After correction and recalibration:

| Comparison | Simulator | Published | Status |
|---|---|---|---|
| Gentamicin CL in SAM | 0.088 L/h/kg | 0.103 (0.02–0.16) | within range |
| Gentamicin Vss in SAM | 0.509 L/kg | 0.670 (0.26–1.31) | within range |
| Gentamicin V1 in SAM | 0.163 L/kg | 0.254 | 1.56× too small |
| Lumefantrine day-7, SAM | 173 ng/mL | 246 (IQR 160–438) | within IQR |
| Lumefantrine day-7, non-SAM | 290 ng/mL | 330 (IQR 216–503) | within IQR |
| Lumefantrine SAM/non-SAM ratio | 0.60–0.74 | 0.745 | fitted to this |
| **Rifampicin severe/adequate AUC** | **0.48** | **~1.00 (n.s., two studies)** | **disagrees** |

**The rifampicin disagreement stands and has not been fitted away.** The
decomposition localises it to plasma-protein binding: hypoalbuminaemia raises
the unbound fraction 29% in the simulator, and the one study that measured
unbound fraction directly found it unchanged. The same pathway also breaks a
directional check on beta-lactam renal clearance, which makes protein binding
the most likely location of the largest remaining error.

Note, however, that Verrest et al.'s systematic review reports *increased*
elimination for albumin-bound drugs including rifampicin in malnutrition,
which is the direction the simulator predicts for the clearance limb. The
disagreement is therefore narrower than a single null AUC result suggests:
what is contested is the magnitude and the binding mechanism, not obviously
the sign. This has not been re-fitted either.

Sources: Seaton et al. 2007 *JAC* 59:681 · Denoeud-Ndam et al. 2016
*BMC Medicine* 14:167 · te Brake et al. 2015 *AAC* 59(6) · Ngcobo 2025
*Clin Pharmacokinet* 64(9):1283 · Verrest et al. 2021 *Clin Pharmacokinet*
60(9):1149.

## Headline results

Full tables, including the failures, are in [`RESULTS.md`](RESULTS.md).

**Accuracy** on held-out patients, against the noise-free simulated curve, from
sparse noisy samples and partially-missing labs: median fold error **1.050**,
**98.6%** within two-fold, R² (log) **0.997**. The latent PK parameters — never
given as inputs — are recovered with R² 0.981–0.999.

**Supervision matters more than architecture.** Under *identical* inputs, *identical* supervision and within 3% the
same parameter count, swapping the mechanistic solver for a direct
curve-regression head costs **1.54×** in log-RMSE (0.705 → 1.087). Adding the
auxiliary supervision on parameters and latent physiology to the hybrid is
worth considerably more: **3.7×** (0.705 → 0.190).

The reason is diagnosable rather than mysterious: softening the
absorption model during external calibration made bioavailability and
clearance harder to separate from concentration data alone, so the
seven-parameter bottleneck became partly unidentifiable without parameter
supervision — a penalty the black box, having no bottleneck to confuse, does
not pay. The solver still earns its place, and still buys the four structural
properties above, which no amount of supervision provides. It is simply not
the largest numerical effect.

**Water chemistry is load-bearing, not decorative.** Removing it costs
**2.05×** (log-RMSE 0.190 → 0.389); removing the whole bedside biochemistry
panel costs **1.34×** (→ 0.254). Anthropometry partly proxies for albumin and
CRP; nothing in the clinical examination proxies for what is in the well. The
redundancy this implies is directly confirmed: once water chemistry is gone,
discarding the laboratory panel as well costs almost nothing further
(2.05× → 2.19×).

**Permutation importance partly recovers the mechanism it was never shown**,
in order: body weight → oedema grade → sex → water arsenic → serum albumin →
water lead. Body size, oedema and the two strongest exposure–response relations
appear where the simulator's construction puts them. Water cadmium falls
outside the top six.

Sex ranking third is **the model over-attributing, and we chased it down**.
Flipping sex in the simulator changes fat-free mass by 3.3% and moves only V1
and V2 (RMS log 0.042); it does not move CL, Q, ka, F or fu at all. Flipping the
same input for the model moves V1 by 0.148 and V2 by 0.198 — 3.5× and 4.8× too
much — and moves all five parameters the simulator leaves untouched. The
attribution method is right about the model; the model has learned a sex
dependence the mechanism does not contain, most likely because four other inputs
(the three z-scores and weight-for-height percentage) are computed against
sex-specific references, so sex is partly recoverable from them and a network
can split a learned effect among redundant inputs however it likes. Auxiliary
supervision on latent physiology did not prevent this, which bounds the
interpretability claim above. Reproduce with `python scripts/diagnose_sex.py`.

So the importance check recovers the causal ordering for the mechanistic
covariates, and exposes an over-attribution for one that the simulator barely
uses.

**Model-guided vs weight-band dosing**, on a fresh population of 120 virtual
children, with the simulator scoring both:

| | standard mg/kg | model-guided |
|---|---|---|
| reached PK/PD target | 69.1% | **87.5%** |
| over-exposed | 1.6% | **0.0%** |

Per drug: rifampicin 8% → 88%, albendazole 62% → 98%, isoniazid 74% → 99%,
while lumefantrine, ceftriaxone and fluconazole already attain their targets
and are left alone. Gentamicin keeps 100% attainment and drops from **12.5% to
0%** over-exposed. The median dose ratio stays at 1.00× for
most drugs — the model leaves alone the children it is already serving and
moves only the ones it is failing. In severe oedematous malnutrition the
10th–90th percentile dose range runs 1.00× to 4.00×, which is the whole
argument against a single weight band in one number.

**The targets determine several of these numbers, which is itself the finding.**

*Over-exposure depends entirely on one threshold.* Gentamicin's trough
threshold was 2.0 mg/L and unsourced. The published once-daily paediatric
target is ≤0.5 mg/L (≤1.0 for neonates): Hollander et al., *Antibiotics*
2023;12(5):810.

| gentamicin trough threshold | over-exposed, label | over-exposed, model |
|---|---|---|
| 2.0 mg/L (unsourced) | 0.0% | 0.0% |
| 1.0 mg/L (neonatal) | 2.5% | 0.0% |
| **0.5 mg/L (paediatric, used here)** | **12.5%** | **0.0%** |

*Amoxicillin and rifampicin targets are not met in a well-nourished child.*
Amoxicillin targets 50% fT>MIC but a well-nourished reference child reaches
0.29 at label dose; rifampicin targets free AUC/MIC 38 and reaches 24. For
those two drugs the low baseline attainment is not attributable to
undernutrition. At a 40% amoxicillin target the same model gives **19% → 72%**
(`outputs/ci_amox_target40.json`). Isoniazid runs the other way: its target of
14 is met roughly sixteenfold in a well-nourished child, against published
values of 567–707, so its improvement measures very little.

When *no* safe dose reaches the target, the tool says so rather than returning
a number: it reports the most exposure that stays safe and names the binding
mechanism. For a severely oedematous child on rifampicin it infers F = 0.15
and states that the patient is absorption-limited, so escalating the oral dose
cannot fix it — which is the clinically actionable finding, and one a
dose-only recommender would have hidden.

**Where it fails, and why.** Held out an entire *drug*, the model does not
transfer: median fold error **11.2**, 17.8% within twofold. Gentamicin is the
worse of the two held-out drugs (fold error 19.4, 8.7% within twofold) and the
reason is legible — remove it and no training drug is left that is at once
intravenous, essentially unbound and ~100% renally cleared. Rifampicin, which
has close analogues in the training set, degrades to 4.8-fold instead: poor,
but recognisably in the right regime. Transfer to a new drug works only by
interpolation inside covered chemical space, and a nine-drug library does not
cover much.

**Uncertainty.** The residual-error model is well calibrated out of the box
(nominal 90% interval covers 89.3% of observations). Monte-Carlo dropout is
not: its nominal 90% interval covers 45.0% of the truth. One variance-scaling
scalar fitted on validation (3.13×) brings that to 86.0%. This is not cosmetic
— the dose recommender consumes that distribution, and an overconfident one
picks doses with no margin.

## What the simulator reproduces

Dosed at the identical label mg/kg, exposure diverges systematically by
nutritional class — and **in different directions for different drugs**, which
is the reason a single correction factor cannot work:

| Drug | Driver | Severe vs adequate |
|---|---|---|
| rifampicin | bile-salt-dependent absorption + CYP3A4/SULT loss | ↓↓ ~4-fold |
| lumefantrine | extreme lipophilicity, needs bile and fat | ↓↓↓ |
| albendazole | poor solubility + high first-pass | ↓↓ |
| amoxicillin | reduced clearance, ECF-tracking volume | ↑ slightly |
| gentamicin | pure ECF marker, pure renal clearance | ↑ (oedema) |
| ceftriaxone | hypoalbuminaemia raises fu → faster clearance | ↓ |

The rifampicin and lumefantrine signals are the well-documented clinical
problem this model exists to represent.

---

## Honest limitations

These are not boilerplate. They bound what the results mean.

1. **The model is trained on synthetic data.** It is a surrogate for the
   simulator, and it is only as right as the simulator is. Its accuracy
   numbers measure how well it emulates a set of equations, **not** how well it
   predicts a child.
2. **The simulator is partly calibrated and nowhere validated.** Most constants
   were chosen for plausibility. Three sets are now anchored to published
   measurement (gentamicin reference PK, the toxicant potency scale, the
   enteropathy absorption coefficients) — but that rests on three drugs and
   four sources, one a review and one in adults. Six of nine drugs have never
   been compared against data.
3. **The coupling between nutrition and toxicant handling is the least
   sourced part of the model, and it is the project's distinctive claim.**
   Reviewing the literature for each mechanism found the general pharmacology
   well supported and the nutrition-dependent limb repeatedly unsupported:
   SAM is the methyl donor for arsenic methylation, but methyl-donor
   *limitation under protein deficit* has no source here; iron deficiency
   raises DMT1 and cadmium uptake in rats, but the extension to lead,
   manganese and to children does not; α1-acid glycoprotein is a positive
   acute-phase protein that binds basic drugs, but the resulting fall in free
   fraction is an inference. Cadmium raises urinary tubular-injury markers but
   a reduction in filtration is unsourced. The organ-sparing exponent of 0.45 —
   the mechanism external validation found inert and then repaired — likewise
   has no measured source.
4. **Several PK/PD targets determine their own results.** Five of nine are now
   sourced (beta-lactams to Craig 1998; gentamicin to Hollander 2023;
   lumefantrine to WWARN 2015; fluconazole to EUCAST 2008, though Andes gives a
   fourfold looser value). Four are not. Gentamicin's threshold moves
   over-exposure from 0% to 12.5%; amoxicillin's moves attainment from 15% to
   72%; rifampicin's is not attained by a well-nourished child; isoniazid's is
   roughly fortyfold below published values. Paracetamol carries a placeholder
   AUC/MIC target with an MIC, meaningless for an analgesic; it is excluded from
   the dosing experiment and no reported result depends on it, but it should be
   replaced. See `REFERENCES.md`.
5. **No real-world validation exists here.** The honest path to use would be:
   fit the simulator's parameters to real sparse PK from malnourished cohorts,
   re-derive the model, and prospectively validate target attainment. Steps
   one and three have not been done.
6. **Structural assumptions.** Linear, time-invariant PK throughout. No
   saturable binding (which matters for ceftriaxone), no enzyme
   auto-induction (which matters for rifampicin), no metabolite tracking (so
   paracetamol's NAPQI pathway is represented only through its parent),
   no transporters, no drug-drug interactions, no nutritional recovery over
   time.
7. **The toxicokinetics is a steady-state approximation.** Real burdens depend
   on intermittent exposure, growth dilution and bone remodelling.
8. **The target attainment improvements reported by
   `scripts/clinical_impact.py` are measured against the same simulator that
   generated the training data.** They demonstrate that the model has
   successfully inverted the mechanism it was shown. They are an upper bound
   on, and not evidence of, real-world benefit. The 69.1% → 87.5% figure says
   the model learned the simulator, not that it would help a child.
9. **The model does not transfer to an unseen drug** (above). Anything outside
   the nine-drug library is out of scope.
10. **MUAC, WHZ and oedema grade are measured with error in the field** at rates
    likely higher than the ~1.2% assumed here.

## Attribution of the physiology

Functional forms are motivated by the literature on malnutrition physiology
and paediatric PBPK: organ sparing in undernutrition; extracellular water
expansion with reciprocal intracellular depletion in oedematous malnutrition;
IL-6/CRP-mediated down-regulation of CYP3A4 with relative sparing of CYP2E1;
hypoalbuminaemia raising unbound fraction of acidic drugs while AGP rises as a
positive acute-phase reactant; environmental enteric dysfunction (villous
blunting, raised lactulose:mannitol ratio, hypochlorhydria, bile-salt
deficiency); the upward bias of creatinine-based GFR estimates in low-muscle-mass
patients, which motivates the kidney-mass-based GFR model used here; DMT1
up-regulation in iron deficiency increasing divalent-metal uptake; and
methyl-donor limitation of arsenic methylation under protein deficit.

See `PROVENANCE` at the end of `malnupk/physiology.py`, and `REFERENCES.md`
for which of these claims currently have a verified source and which do not.

## Citing this work

```bibtex
@article{manne2026malnupk,
  title   = {Mechanism-constrained deep learning recovers individual pharmacokinetics
             under undernutrition and water-toxicant co-exposure: an in silico study},
  author  = {Manne, Pranav},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/pranavmanne17/malnupk}
}
```

`CITATION.cff` carries the same metadata in a form GitHub and Zenodo read
directly.

## License

Code is released under the [MIT License](LICENSE). The manuscript and figures in
`paper/` are intended for release under CC BY 4.0 with the preprint.

**Not a medical device.** No warranty. Nothing here should inform the care of
any patient.
