# Theoretical Framework — Capability Approach + Critical Data Studies

## 1. Why this is not a digital-development index

There is a recognisable family of digital-development indices: ITU's IDI, WEF's NRI, World Bank's GovTech Maturity Index, UN's EGDI, BCG's Digital Inclusion Index, and the World Bank DPI Composite. They share a structure: pillars × indicators → weighted aggregate → cross-country ranking. They are useful and well-instrumented. They are also, with few exceptions, **resource-and-readiness indices**: they measure whether infrastructure, institutions, and skills *exist*, not whether people are *substantively free* to use them.

A capability-grounded index is a different object. It commits to a normative position about what counts. It refuses to treat the existence of a service as evidence that people can access it. It demands attention to **conversion factors** — the personal, social, and environmental conditions that mediate the move from resource to capability. It is suspicious of single-number rankings.

DCLO, as the name promises, should be the second kind of object. This document defends that position and translates it into specific operational commitments.

## 2. The capability approach in five claims

Following Sen (1985, 1999), Nussbaum (2000, 2011), and Robeyns (2017):

1. **The evaluation space matters.** What we measure to evaluate well-being is not a neutral choice. Utility, primary goods, resources, and capabilities are different evaluative spaces; the choice between them is political. The capability approach evaluates in the space of capabilities — the substantive freedoms a person has reason to value.
2. **Resources are not capabilities.** Two people with the same income or the same broadband connection may have very different capabilities to use them, depending on conversion factors (disability, gender, language, social norms, infrastructure quality, time burden, fear).
3. **Capabilities are plural and incommensurable.** A capability index that collapses everything into one number imposes a choice of trade-offs that the capability approach itself rejects (Sen 1999; Robeyns 2017, Ch. 2). Aggregation, when it happens, must be defended.
4. **Functionings are achievements, capabilities are options.** Two people can have the same achieved functioning (e.g., banked) by very different routes — one freely chosen, one coerced by lack of alternatives. The freedom matters, not just the achievement.
5. **The list of capabilities to attend to is not given.** Nussbaum proposes a fixed list of ten central capabilities; Sen rejects a fixed list and insists on democratic deliberation. The applied-measurement literature (Anand et al. 2009; Alkire & Foster 2011) has tried to instantiate both.

## 3. Translating the capability approach to digital systems

Adapting Mansell (2017), Helsper (2021), Robinson et al. (2015), and Eubanks (2018), I propose the following digital-capability schema:

### 3.1 Distinguish four layers

| Layer | What it asks | Indicator class |
|---|---|---|
| **Resources** | What infrastructure, devices, services, data flows are present? | Coverage, penetration, station count, points of service. |
| **Conversion factors** | What allows a person to convert presence into use? | Skills, language, gender norms, disability, affordability, time, trust. |
| **Capabilities** | What is a person substantively free to do or be online? | Use a service of their choice, transact safely, find work, exercise voice, refuse without penalty. |
| **Functionings** | What do people actually do or achieve? | Bank account is used, child finishes online schooling, health appointment is kept. |

The capability layer is the index's evaluation space. The four-layer structure prevents the common failure of presenting a resource indicator (e.g., "secure internet servers per million") as if it measured what people can do.

### 3.2 Six domains, redefined as capability sets

Where the current build labels six domains (ACC, SKL, SRV, AGR, ECO, OUT), I redefine each as a **capability set** — a bundle of substantive freedoms — and indicate which of the four layers each indicator should target.

| Domain | Capability set (Sen-shaped) | Layer mix | Reference Nussbaum capabilities |
|---|---|---|---|
| **ACC — Connectivity** | The freedom to be reachable and to reach others through digital networks at affordable cost and without coerced exposure. | Resources + conversion factors. | Affiliation; Control over one's environment. |
| **SKL — Digital literacy and competence** | The freedom to use digital tools with effective competence for personally chosen ends. | Conversion factors + capabilities. | Senses, imagination, thought; Practical reason. |
| **SRV — Access to essential services through digital channels** | The freedom to obtain health, education, finance, and government services through digital channels of one's choice, at no cost to dignity. | Resources + capabilities + functionings. | Bodily health; Education; Affiliation. |
| **AGR — Agency, safety, rights** | The freedom to act online without fear; to consent or refuse; to be free of surveillance and platform coercion; to exercise voice. | Capabilities. | Bodily integrity; Affiliation; Control over one's environment; Practical reason. |
| **ECO — Economic participation** | The freedom to participate in the digital economy on terms that do not require accepting precarity, surveillance, or wage theft. | Conversion factors + capabilities. | Affiliation; Control over one's environment; Other species (in the limit). |
| **OUT — Realised functionings (achievements)** | The achieved functionings (not the freedoms): banked, employed, educated, healthy, civically participating *via* digital channels. | Functionings. | All. |

Two consequences follow. First, **OUT is not an "outcome" domain in the macroeconomic sense**; it is a functionings domain. The current build's use of WB_AG.LND.AGRI.ZS and WB_SP.POP.GROW under OUT is wrong. Second, **AGR and ECO are not residuals**; they are the heart of the capability claim, because that is where the freedom-vs-coercion distinction actually bites.

### 3.3 The conversion-factor layer is non-negotiable

Helsper (2021) and Hilbert (2011) document four levels of digital divide: motivation, access, skills, outcomes. The "outcomes" divide — that two people with the same access and the same skills get systematically different outcomes — is exactly what capability-approach conversion factors predict. A capability-grounded index has to surface this gap.

Operationally, this means at least one of the following must be present in every release:

- a gender decomposition of every domain;
- an urban–rural decomposition of every domain;
- a within-country inequality penalty (Atkinson-style) on the aggregate.

The current build does none of these.

## 4. Critical data studies — why the index is also an artefact

The capability approach gives a normative grounding. It does not, on its own, force the project to interrogate the politics of measurement. For that, I draw on critical data studies and the political economy of indicators.

### 4.1 Indicators as governance technology

Merry (2016) and Bhuta et al. (2018) show how indicators become governance technologies: they shape what is countable, who is rendered legible, and which interventions become "evidence-based." A digital-capability index is no exception. Naming this is a precondition for building one responsibly.

### 4.2 The DPI critique

India's DPI stack — Aadhaar, UPI, India Stack, ABDM, ONDC — has produced an unusually rich critical literature. Khera (2022) and Rao (2019) document Aadhaar's exclusionary failures. Masiero (2023) places DPI in a political-economy frame of "data-as-development." Krishna (2024) examines who gains and who loses from UPI saturation. Sarkar's own work on infrastructure and informality is in the same lineage.

A paper that benchmarks countries on a capability-shaped digital index without engaging this literature would be ahistorical. The framing here is therefore explicit: **the index is a critical instrument**, designed to surface where DPI rhetoric exceeds DPI capability and where standard indicators fail to register exclusion.

### 4.3 Datafication and "the costs of connection"

Couldry & Mejias (2019, 2024) frame contemporary digital infrastructure as **data colonialism**: extraction of data from the social as a new factor of production. Zuboff (2019) makes the related case for surveillance capitalism. These framings change how we read AGR and ECO indicators. "Digital agency" is not the same in a regime where every transaction is logged and every payment-history record is sold to credit-scoring firms. The index does not have to subscribe to either framing; it does have to be aware of them.

### 4.4 The reflexive turn

Suchman (2007) and Eubanks (2018) press measurement projects to ask: *whose questions does the index answer; whose silence does it preserve; whose harm does it normalise.* For DCLO, those questions are:

- Whose questions? Predominantly central-government and donor questions ("how digitally ready is the country?"). Less so frontline-user questions ("can I trust this app with my medical record?").
- Whose silence? Indicators do not register people without official identity, women whose accounts are male-controlled, gig workers whose digital participation is forced.
- Whose harm? A poorly-ranked country may be more, not less, attentive to dignity in service design. A high rank can normalise coerced inclusion.

These are not afterthoughts. They are why the dashboard must report constraints and ranks together, and why the paper has to frame the ranking as **a partial visibility**, not a verdict.

## 5. The estimand

If the project is committed to capability framing, the estimand of the causal layer is not "the effect of DCLO on service outcomes." It is something narrower and defensible: **the within-country, within-period association between prior-period digital-capability domain scores and prior-period-controlled subsequent functioning achievement, conditional on time-invariant country structure and global shocks, and bounded by indicator-construct validity.** The bound is doing the work — without it, the estimand is over-specified.

In Pearlian terms: the DAG (see `04_identification_strategy_revised.md`) specifies that DCLO at *t-1* and the functioning at *t* share a backdoor path through (a) GDP per capita, (b) urbanisation, (c) state capacity. Two-way fixed effects close (a) and (b) only to the extent they are time-invariant. They are not. The TWFE estimate is therefore a within-entity association under explicit unconfoundedness assumptions; the paper must say so.

## 6. Operational commitments derived from this framework

1. **Domain re-justification.** Every domain in the next release must include at least one indicator from the *capability* or *functioning* layer; resources-only domains will be flagged.
2. **Conversion-factor surfacing.** Gender and urban-rural decomposition for at least three domains in the India track; a within-country inequality penalty for the aggregate where data permit.
3. **Construct-validity gates.** Indicator inclusion requires (a) a documented mechanism by which the indicator captures the targeted layer, (b) coverage > 0.40 of panel cells (not just per-indicator pct_observed), (c) absence of a category error against the layer.
4. **Reflexive caveats in every release.** Inclusion list, exclusion list, what the index does not see, named in the dashboard explainer and the paper.
5. **Critical instrument framing.** The paper foregrounds the index as a diagnostic device for testing whether the standard digital-development indicator family captures capability. Where it does not, the paper says so.

## 7. References (working set; full file in `bibliography.bib`)

- Sen, A. (1985). *Commodities and Capabilities*. North-Holland.
- Sen, A. (1999). *Development as Freedom*. Oxford UP.
- Nussbaum, M. (2000). *Women and Human Development*. Cambridge UP.
- Nussbaum, M. (2011). *Creating Capabilities*. Harvard UP.
- Robeyns, I. (2017). *Wellbeing, Freedom and Social Justice: The Capability Approach Re-Examined*. Open Book.
- Anand, P., et al. (2009). The development of capability indicators. *Journal of Human Development and Capabilities*, 10(1), 125–152.
- Alkire, S., & Foster, J. (2011). Counting and multidimensional poverty measurement. *Journal of Public Economics*, 95(7–8), 476–487.
- Mansell, R. (2017). *Imagining the Internet: Communication, Innovation, and Governance*. Oxford UP.
- Helsper, E. J. (2021). *The Digital Disconnect: The Social Causes and Consequences of Digital Inequalities*. Sage.
- Hilbert, M. (2011). The end justifies the definition: The manifold outlooks on the digital divide and their practical usefulness for policy-making. *Telecommunications Policy*, 35(8), 715–736.
- Robinson, L., et al. (2015). Digital inequalities and why they matter. *Information, Communication & Society*, 18(5), 569–582.
- Merry, S. E. (2016). *The Seductions of Quantification*. Chicago UP.
- Bhuta, N., Malito, D. V., & Umbach, G. (Eds.) (2018). *The Palgrave Handbook of Indicators in Global Governance*. Palgrave.
- Khera, R. (2022). *Dissent on Aadhaar*. Orient Black Swan.
- Rao, U. (2019). Population meets database: aligning personal, documentary and digital identity in Aadhaar-enabled India. *South Asia*, 42(3), 537–553.
- Masiero, S. (2023). Should we still be doing ICT4D? *Information Systems Journal*, 33(5), 1–10.
- Krishna, S. (2024). The political economy of digital public infrastructure. *Information Society*, forthcoming.
- Couldry, N., & Mejias, U. A. (2019). *The Costs of Connection*. Stanford UP.
- Mejias, U. A., & Couldry, N. (2024). *Data Grab*. WH Allen.
- Zuboff, S. (2019). *The Age of Surveillance Capitalism*. Public Affairs.
- Eubanks, V. (2018). *Automating Inequality*. St. Martin's.
- Suchman, L. (2007). *Human–Machine Reconfigurations*. Cambridge UP.
- Apaza, C. R. (2009). Measuring governance and corruption through the worldwide governance indicators. *PS: Political Science & Politics*, 42(1), 139–143.
- Thomas, M. A. (2010). What do the worldwide governance indicators measure? *European Journal of Development Research*, 22(1), 31–54.
- Langbein, L., & Knack, S. (2010). The worldwide governance indicators: six, one, or none? *Journal of Development Studies*, 46(2), 350–370.
- Christensen, G., & Miguel, E. (2018). Transparency, reproducibility, and the credibility of economics research. *Journal of Economic Literature*, 56(3), 920–980.
