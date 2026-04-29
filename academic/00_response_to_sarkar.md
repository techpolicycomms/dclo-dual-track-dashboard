# Response to Prof. Swagato Sarkar's suggestion: "treat the dashboard as a paper"

**Author:** Rahul Jha, JSGP, O.P. Jindal Global University
**Supervisor:** Prof. Swagato Sarkar, JSGP, OPJGU
**Date:** 2026-04-29
**Status:** internal working document; intended to seed a supervision conversation, not for circulation.

## 1. The suggestion as I understood it

In supervision, you said the dashboard could itself be a paper if I held it to academic discipline rather than treating it as a product. I read that as four directives, not one:

1. **Make the conceptual claim explicit.** The dashboard is named *Digital Capability for Life Outcomes*. That phrasing is not decorative — it commits the project to Sen's capability approach. A paper has to defend the capability framing, not just borrow the vocabulary.
2. **Make the empirical claim falsifiable.** A composite score on its own is a description. A paper needs an estimand, a population, a counterfactual, and a way of being wrong.
3. **Make the political economy visible.** Indices rank entities. Ranking is a political act. A paper from JSGP cannot pretend ranking is neutral; it has to interrogate who is served by the ranking, who is rendered legible, and on whose data.
4. **Make it reproducible to a level a reviewer would accept.** Open data, locked code, documented decisions, pre-registered analysis where feasible.

This document is my honest assessment of where each directive currently stands.

## 2. Directive-by-directive status

### 2.1 Conceptual claim: capability framing

**Where the build complies.** The constructed score is named DCLO and the methodology document (`docs/methodology-formative-sem.md`) treats it as a formative multidimensional construct in the Bollen–Lennox / Diamantopoulos–Winklhofer / Jarvis et al. tradition. Six SDG-aligned domains. Equal weights at baseline. Sensitivity scenarios for weights.

**Where it does not.** The methodology document cites measurement-theory references (Bollen 1991; Diamantopoulos 2001; Jarvis 2003; Hair 2012; Petter 2007) but **does not cite Sen, Nussbaum, or Robeyns**. The construct is described as a "capability for life outcomes" but the underlying logic is **measurement-as-aggregation**, not **capability-as-substantive-freedom**. The capability approach requires:

- a distinction between **resources** (what people have access to), **conversion factors** (what allows them to convert that access into doings/beings), **capabilities** (what they are substantively free to do or be), and **functionings** (what they actually do or are);
- explicit attention to the **evaluation space** (we measure capabilities, not utility, not GDP-equivalents, not preference satisfaction);
- a **plural-and-incommensurable** treatment of dimensions (resisting collapse into one number when the dimensions are not substitutes).

DCLO as currently built does none of this. ACC ("Access and Connectivity") is a resource indicator; SKL is a resource indicator (enrolment); SRV is a state-capacity-of-services indicator; OUT in the country track is a set of macroeconomic indicators (population growth, agricultural land, clean cooking access). These are not capabilities. They are inputs and aggregate stocks.

This is fixable, and `02_theoretical_framework.md` and `03_indicator_validity_audit.md` describe how. But it is the most important gap and you would be right to flag it.

### 2.2 Empirical claim: falsifiability

**Where the build complies.** A causal layer exists: two-way fixed effects with cluster-robust SEs, lagged predictors, a permutation placebo, a small robustness battery, and rank-stability under Dirichlet weight perturbation. Output is published. The method is documented in `docs/dclo-causal-methodology.md` and `docs/dclo-identification-strategy.md`. The audit-logger writes input/output SHA-256, environment, and seeds. This is more than most index papers do.

**Where it does not.** Three things.

First, the headline result — `DCLO_score(t-1) → SRV_score(t)` with β ≈ 0.62, p < 0.001 — is **not a capability claim**. SRV is built from four WTO services-export proxies (`WTO_SERVICES_TOTAL_EXPORTS`, `_TRANSPORT`, `_TRAVEL`, `_POSTAL_COURIER`). These are macroeconomic services-trade indicators. They tell us nothing about whether a person in Nashik or Nagaland can access an e-health service or finish a school-leaving exam through a digital channel. Yet they are the dependent variable on which the entire causal narrative rests. The pooled robustness recovers β ≈ 1.0, R² = 0.91 — a giveaway that predictor and outcome are largely the same composite of macroeconomic stocks.

Second, the **method-comparison table** (`data/gold/dclo_method_comparison.csv`) reports Spearman ρ between the baseline rank, the confidence-weighted rank, and the causal-signal rank. ρ between baseline and causal flips from −0.55 in 2016 to +0.76 in 2024. This is reported in the dashboard as a routine "method agreement over time" view; it should be reported as **a stability failure**.

Third, the placebo passes too easily. Permuting `DCLO_score` within year and re-fitting yields β ≈ −0.014, p ≈ 0.12. Good. But **a within-year permutation breaks the cross-sectional structure entirely**. A more disciplined falsification is a **leave-future-out placebo**, an **event-study around UPI/India-Stack rollouts**, or a **placebo outcome** (e.g. agricultural land share, which should not respond to digital capability). None of these are implemented.

### 2.3 Political-economy lens

**Where the build complies.** The dashboard exposes a "model trust tier" derived from how many domains are covered for an entity-year, and tags low-trust observations. It logs imputation share. It reports rank uncertainty. The state-track surfaces context adjustment via a DPI composite.

**Where it does not.** Nowhere does the build interrogate:

- **Who is rendered legible by the index and who is not.** Comparators are 47 large economies (typical DPI panel). Small island states, low-income African states, and several South Asian peers do not appear, because the WB-DPI panel does not cover them. Saying "this is the comparative track" without naming this exclusion is a category error in a JSGP-supervised paper.
- **Whose data is the basis for the AGR ("Agency, Safety, Rights") domain.** The current build operationalises agency through Worldwide Governance Indicators (Control of Corruption, Political Stability), which are themselves expert-coded composites contested in the political-science literature (Apaza 2009; Thomas 2010; Langbein and Knack 2010). Reducing **digital agency** to a percentile rank produced by a different consortium for a different purpose is exactly the kind of indicator laundering Merry (2016) and Bhuta et al. (2018) warn against.
- **The relation between the index and Indian DPI (Aadhaar, UPI, India Stack)**. The literature on this is now substantial and politically charged (Khera 2022; Masiero 2023; Rao 2019; Krishna 2024). DCLO is, in effect, scoring the success of a particular state-data infrastructure project. To do that without engaging the critique is a defensible methodological choice only if it is named as such.

This is the single biggest difference between "good engineering documentation" and "a paper from JSGP." The technical artefacts are fine; the *standpoint* is missing.

### 2.4 Reproducibility

**Where the build complies.** Audit logger writes full provenance: input/output SHA-256, environment, package versions, seeds, parameters, stage-level row accounting. Pipelines are CLI-callable with a config file. Verification reports are published as JSON next to the gold tables. There is a knowledge-base wiki. The dashboard exposes all of this in a "Data Provenance & Audit" tab. This is unusually good for a single-author project.

**Where it does not.** No preregistration. No locked dataset version per release. No DOI on the gold tables. No formal data-management plan. No statement of conflicts. No competing-interests declaration. No ethics review note (the project uses public administrative data so probably exempt, but a paper has to say so).

## 3. What this folder commits to

I am going to treat the dashboard as a paper *on these specific terms*:

1. The paper's claim will not be "DCLO predicts service outcomes." It will be **"a capability-grounded reading of currently-feasible DPI indicators reveals which countries' DPI investment narratives are empirically supported and which are not — and which capabilities the standard index family ignores."** This is a defensible, JSGP-shaped paper. The dashboard becomes a methodological contribution and a critical instrument, not an oracle.
2. The paper will report the headline TWFE result as the **failure mode it is**, not the headline finding it currently is.
3. The paper will integrate the political-economy critique by name (Sarkar's own work on infrastructure and labour; Khera; Masiero; Bhuta et al.; Merry).
4. The paper will pre-register the next analysis run before the 2026 data refresh.

`08_paper_outline.md` and `09_paper_draft.md` instantiate this commitment.

## 4. Asks

1. **A 30-min check on the framing.** Does the "capability-grounded reading + critique of DPI indicator orthodoxy" framing land? If you would rather lead with the political-economy critique and put the index second, the structure of the paper changes substantially — `08_paper_outline.md` has both options.
2. **Pointer to your own work on infrastructure** that you would like cited or engaged. I have included generic references in `bibliography.bib` but it would be wrong to gesture at your line of work without naming the specific papers you'd want me to read.
3. **A view on co-authorship.** JSGP norms allow co-authorship on supervised work. I would prefer to know your preference early so I can write to that level of accountability.
4. **A view on venue.** `10_target_venues.md` ranks options. My current default is *Information, Communication & Society* (Q1, IF ~7) for the critical-instrument framing, with *Big Data & Society* as backup. If you want me to target a development-economics venue (*World Development*, *Journal of Development Economics*) the methodological centre of gravity has to shift.

## 5. Honest sentence

The dashboard exists. The data pipeline is solid. The audit trail is good. But the paper does not yet exist, because the conceptual claim, the empirical claim, and the political-economy claim have not been forced to confront each other. This folder is the work of forcing that confrontation.
