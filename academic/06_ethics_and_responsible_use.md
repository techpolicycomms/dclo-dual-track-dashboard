# Ethics and Responsible Use

A digital-capability index that ranks countries — and, in the state-track, ranks Indian states — is a **governance technology**, in Sally Engle Merry's sense (Merry 2016). This document states the ethical commitments under which DCLO is published, the harms a ranking can cause, and the reflexive caveats that accompany every release.

## 1. Sources of risk

### 1.1 Legitimising data extraction

A high DCLO score in the current build is, in part, a high WGI / high-services-trade / high-WB-internet-penetration score. Treating that as "high digital capability" can legitimate rapid DPI build-out at the cost of consent, voice, and refusal — which are precisely the things AGR (Agency, Safety, Rights) ought to measure but currently does not (`03_indicator_validity_audit.md`). The risk is that the index becomes a marketing instrument for the kind of state-data infrastructure project Khera (2022), Masiero (2023), and Krishna (2024) document as exclusionary in practice.

**Commitment.** The next release implements the AGR replacement set (Freedom House FOTN; SDG 16.10.2; Privacy International data-protection laws status). Until then, the dashboard surfaces an explicit caveat naming the construct error.

### 1.2 Rendering some populations invisible

The country panel includes 47 economies, most of them upper-middle and high-income. Several South Asian peers, most low-income African states, and small island developing states are absent — not because they have no digital capability, but because the indicator panel does not register theirs. Saying "the comparative track" without naming this exclusion is a category error.

**Commitment.** Every release publishes an inclusion table and an exclusion table with explicit reasons. The paper foregrounds this as a known limitation of the standard indicator family.

### 1.3 Aggregation disguising distributional harm

A composite score is a population mean of (means of) z-scored indicators. If a country has high mean digital capability and a long tail of digitally-excluded households, the mean rewards the country. Capability theory (Robeyns 2017) explicitly rejects aggregation that masks inequality.

**Commitment.** Every release that uses an aggregate also reports either (a) gender decomposition, (b) urban–rural decomposition, or (c) an Atkinson-style inequality penalty for at least three domains. Where data do not yet permit, the absence is named.

### 1.4 Indicator laundering

Re-using indicators built for other purposes (WGI for governance; WTO services exports for trade) in an index whose name promises something else is a form of indicator laundering (Bhuta et al. 2018). It transfers the legitimacy of the source onto a claim the source was not designed to support.

**Commitment.** Each indicator carries a `targeted_layer` field (resources / conversion factors / capabilities / functionings) and a `mechanism_paragraph` justifying its inclusion. Indicators without both are excluded.

### 1.5 Misuse of rankings

A rank-ordered list invites comparative competition (which country improved? which fell?) without supporting it. Rank changes within ±2 in a panel where panel-coverage is 8 % are not interpretable. The current dashboard reports rank changes without that bound.

**Commitment.** The dashboard surfaces rank uncertainty (already present via Dirichlet stability draws); the next release will additionally **suppress** display of rank changes smaller than the rank-uncertainty band.

### 1.6 The reflexive turn

The author is a PhD researcher at JSGP (Jindal School of Government and Public Policy, OPJGU) studying digital infrastructure under the supervision of Prof. Swagato Sarkar, whose research engages the political economy of infrastructure and labour. The author has no commercial funding, no consulting role with any government or platform, and no financial interest in any DPI vendor. The author's ICT-policy perspective is shaped by ITU work and prior development practice; this is a non-trivial standpoint and is named here to allow the reader to weight the index accordingly.

## 2. Data ethics

- **Personal data.** No personal data are processed. All data are aggregate macro indicators. No individual-level inference is made.
- **Licensing.** Source data are used under their respective licences (World Bank: CC-BY-4.0 for most indicators; UN agencies: CC-BY-3.0 IGO; ITU: subject to ITU data policy; Findex: CC-BY-4.0). Aggregations and derived gold tables are released under CC-BY-4.0.
- **Storage and retention.** Raw and curated tables are versioned in `data/raw/` and `data/curated/`; gold tables are versioned in `data/gold/`. Provenance is recorded by the audit logger (`src/quality/audit_logger.py`).
- **Right to be removed.** Source providers retain ownership; if any provider issues a takedown notice the release is republished without the affected indicator and a discontinuity flag is recorded.
- **State-track district-level data.** When district-level indicators are added (planned), the unit of analysis remains the district, not the household. No re-identification is possible from the gold tables.

## 3. Conflicts of interest

The author declares no conflicts. JSGP supervision is acknowledged. No funder, government, donor, or platform has had editorial control over the construction or publication of DCLO.

## 4. Communication discipline

When DCLO is communicated outside the academic track:

1. **No single-number headline** is presented without the accompanying coverage tier and the construct-validity caveat.
2. **No country-vs-country claim** of the form "country X overtook country Y" is made if the rank gap is within the year's Dirichlet rank-stability band.
3. **No causal claim** is made on the basis of the current build's TWFE coefficient (see `04_identification_strategy_revised.md`, §5).
4. **No DPI policy recommendation** is made in dashboard or paper without an explicit identification of the population the recommendation does and does not cover.

## 5. Standards alignment

This document is aligned with:

- **TOP guidelines** (Nosek et al. 2015) for transparency and openness.
- **Christensen & Miguel (2018)** transparency principles.
- **FORCE11 Joint Declaration of Data Citation Principles**.
- **CARE Principles for Indigenous Data Governance** (where relevant).
- **The GDPR principle of data minimisation** (no individual data are needed and none are stored).
- **Cobbe & Singh (2024)** on the duties of public-sector indicator-makers.

## 6. Acknowledgements of harm

Despite the commitments above, an index is still an index. It will be used in ways the author cannot control. Specifically:

- **A high-rank country** may use DCLO to legitimate further DPI investment that, locally, is exclusionary.
- **A low-rank country** may use a low DCLO score to legitimate top-down DPI imposition framed as "catching up."
- **A regional comparator** may misuse the rank as a rhetorical device in policy debate that the underlying construct cannot support.

The paper will name these risks. The dashboard caveat will name them. The folder reader will see them documented here.

## 7. References

- Merry, S. E. (2016). *The Seductions of Quantification*. Chicago UP.
- Bhuta, N., Malito, D. V., & Umbach, G. (Eds.) (2018). *The Palgrave Handbook of Indicators in Global Governance*. Palgrave.
- Khera, R. (2022). *Dissent on Aadhaar*. Orient Black Swan.
- Masiero, S. (2023). Should we still be doing ICT4D? *Information Systems Journal*, 33(5).
- Krishna, S. (2024). The political economy of digital public infrastructure. *Information Society*, forthcoming.
- Couldry, N., & Mejias, U. A. (2019). *The Costs of Connection*. Stanford UP.
- Eubanks, V. (2018). *Automating Inequality*. St. Martin's.
- Nosek, B. A., et al. (2015). Promoting an open research culture (TOP guidelines). *Science*, 348(6242), 1422–1425.
- Christensen, G., & Miguel, E. (2018). Transparency, reproducibility, and the credibility of economics research. *JEL*, 56(3), 920–980.
- Cobbe, J., & Singh, J. (2024). [public-sector indicator duties — placeholder pending verification].
