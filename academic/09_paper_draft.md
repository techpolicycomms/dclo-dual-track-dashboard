# Digital Capability for Life Outcomes: A Critical Instrument for Reading the Standard Indicator Family

**Rahul Jha**
Jindal School of Government and Public Policy (JSGP)
O. P. Jindal Global University, Sonipat
`rjha11@jgu.edu.in`

**Supervisor.** Prof. Swagato Sarkar, JSGP, OPJGU.

**Status.** First-pass long-form draft, intended for supervisory review and subsequent revision after the pre-registered analysis run scheduled for the next data refresh. Figures and final tables are deferred to the post-refresh draft. All numerical statements draw from the existing build artefacts under `data/gold/` and from the supporting documents in `academic/`.

---

## Abstract

This paper develops a capability-grounded composite indicator — Digital Capability for Life Outcomes (DCLO) — and uses it not as a new ranking but as a diagnostic instrument for reading the standard cross-national digital-development indicator family. The proliferating index landscape (ITU IDI, WEF NRI, World Bank GovTech, UN EGDI, BCG Digital Inclusion, World Bank DPI Composite) shares a structural commitment to measurement-as-aggregation: pillars of observable resources are weighted into a single rank. None of these instruments is grounded in the capability approach (Sen, Nussbaum, Robeyns) in a way that distinguishes resources, conversion factors, capabilities, and functionings; none surfaces the conditions under which the existence of digital infrastructure does or does not translate into substantive freedom. Built from the same indicator universe, DCLO inherits these problems by design. We exploit the inheritance: by reading DCLO against itself we expose where the standard family fails capability validity. The conventional within-entity TWFE result that DCLO at *t–1* predicts a Service Enablement (SRV) outcome at *t* with β ≈ 0.62 (95 % CI [0.43, 0.82], n = 470, 47 entities × 10 years) survives only when the outcome is built from macroeconomic services-trade proxies; its pooled-OLS sibling is β ≈ 1.00 (R² = 0.91), a structural-overlap signature. The Spearman rank correlation between the baseline DCLO ranking and a causal-evidence-weighted ranking flips from −0.55 in 2016 to +0.76 in 2024, a stability failure the dashboard had previously presented as routine method agreement. We argue that the conventional finding is largely a structural-overlap artefact: under construct-validated, predictor–outcome-disjoint redefinition, the expected association attenuates substantially. The paper's contribution is therefore methodological and critical, not rank-substitutive. We document the indicator-validity audit, the revised identification strategy, and the political-economy stakes — and we publish DCLO as the artefact of that critique, not as its supersession.

**Keywords.** Capability approach; digital public infrastructure; critical data studies; composite indicators; digital divide; political economy of indicators.

---

## 1. Introduction

The DPI moment has produced a flood of digital-development indices. The International Telecommunication Union maintains the Digital Development Dashboard and the ICT Development Index. The World Economic Forum publishes the Network Readiness Index. The United Nations releases the E-Government Development Index biennially. The World Bank fields the GovTech Maturity Index and a DPI Composite. The Boston Consulting Group runs a Digital Inclusion Index. National variants proliferate: India's NeGD State eReadiness Reports, the European Commission's DESI, ECLAC's Digital Economy Observatory. Each has a distinct pillar structure, indicator set, and weighting scheme; together they constitute a measurement infrastructure for the governance of digital systems.

These indices exhibit a striking pattern: they mostly disagree at country-rank level and mostly agree at ordinal-trend level. Two countries that swap places in one index are often half a quartile apart in another, but both indices agree that the trend over a decade is upward and that the leaders are the leaders. The disagreement is usually attributed to differences in pillar weighting, indicator availability, and editorial scope. The agreement is usually offered as evidence that the indices are converging on a stable underlying construct.

We propose a third reading. The disagreement is the residue of construct-incompatible measurement choices that no amount of weighting reconciles, because the indices are not measuring the same thing. The agreement is, in part, an artefact of shared indicator inputs (the indices reuse World Bank ICT-penetration series, ITU connectivity series, UN administrative records) that ensures their ordinal trends will move together regardless of any underlying capability claim. Neither reading falsifies the indices; both fall well short of the claim that the family measures *digital capability* in any sense the capability approach would recognise.

The capability approach (Sen 1985, 1999; Nussbaum 2000, 2011; Robeyns 2017) holds that the right evaluative space for human well-being is not utility, primary goods, or resources, but *capabilities* — the substantive freedoms a person has reason to value. Capabilities are distinct from *resources* (what a person has access to) and from *functionings* (what they actually do or are). The distinction matters because two people with the same resources may have radically different capabilities, depending on *conversion factors* — disability, gender, language, infrastructure quality, social norms, time burden, fear. None of the standard digital indices operationalise this distinction. They do not separate resources from capabilities; they do not surface conversion factors; they collapse plural-and-incommensurable dimensions into single numbers; and they treat the existence of an institution or service as evidence of substantive freedom to use it. What does it mean, then, to call a country "digitally capable"? On the standard family's own measurement-theoretic terms, it means the country has built more digital things than its peers. That is a worth-knowing fact. It is not a capability claim.

This paper develops a composite indicator — Digital Capability for Life Outcomes (DCLO) — using the same publicly available cross-national indicator universe the standard family draws from. DCLO inherits the family's construct problems by design: this is the methodological commitment. We exploit the inheritance to read the standard family against itself. The questions are three. **(Q1)** Can a capability-grounded composite be built from currently available indicators, and what does it look like when one is attempted? **(Q2)** Where, specifically, does the standard indicator family fail capability validity, and which substitutions does the capability framing demand? **(Q3)** What does the answer to Q2 tell us about how DPI policy is being read through the indicator family — and about who is rendered visible, and who is rendered invisible, by that reading?

The contribution of the paper is critical and methodological. We are not proposing a new ranking. We are proposing a way to use the existing indicators *as a diagnostic surface*: to map where the standard family fails capability, where its causal claims are mechanically over-determined by indicator overlap rather than by behavioural channels, and where its silences fall. The paper's empirical centre of gravity is therefore not "DCLO causes service outcomes" but "the within-panel TWFE coefficient on lagged DCLO is contaminated by predictor-outcome shared content, and this contamination is itself the diagnostic finding." The headline number — β ≈ 0.62 — is reported, but the reading is that it is a structural-overlap signature, not a capability estimate.

The political-economy stakes follow. India's DPI stack — Aadhaar, UPI, India Stack, ABDM, ONDC — has produced a large critical literature (Khera 2022; Masiero 2023; Rao 2019; Krishna 2024). It documents exclusion-by-design, surveillance asymmetries, and the redistribution of risk from state to citizen. Cross-national digital indices score the build-out of the very infrastructures this literature critiques. Doing that scoring without engaging the critique is a defensible methodological choice only if it is named as such. Following Sarkar's research community at JSGP, this paper names it. The index is not neutral. The ranking is not innocent. The aggregation is not the construct.

The rest of the paper proceeds as follows. §2 sets out the theoretical framing — capability approach, critical data studies, and the DPI-critique literature — and positions DCLO as a critical instrument rather than a verdict. §3 describes the construct architecture across six redefined domains and reports the indicator-by-indicator validity audit, including the explicit category errors in the current build. §4 reports the empirical comparison with the standard indicator family. §5 walks through what the headline TWFE result actually says, treating the pooled-vs-FE gap and the method-comparison flip as the diagnostic findings the paper foregrounds. §6 sets out the revised identification strategy. §7 discusses ethics and reflexivity. §8 reads the implications for the broader research programme. §9 names limitations. §10 concludes.

## 2. Theoretical framing

### 2.1 The capability approach in five claims

Five claims, drawn from Sen (1985, 1999), Nussbaum (2000, 2011) and Robeyns (2017), do the framing work for what follows.

First, the *evaluation space* matters. Whether we evaluate well-being in the space of resources, utility, primary goods, or capabilities is not a neutral choice; it is a normative one. Resourcist evaluations privilege the goods one has access to; utilitarian evaluations privilege subjective satisfaction; the capability approach privileges substantive freedoms. Each commits the evaluator to count differently and to value differently.

Second, *resources are not capabilities*. Two people with the same income, the same broadband connection, the same smartphone may have very different capabilities to use them. Capability depends on *conversion factors* — personal (disability, language proficiency, gender, age), social (norms, discrimination, fear), and environmental (infrastructure quality, supply-side reliability, weather). A digital-capability measure that ignores conversion factors is not, in Sen's sense, measuring capability.

Third, *capabilities are plural and incommensurable*. A single capability is not, in general, substitutable for another: more bodily health does not compensate for less affiliation, more practical reason does not compensate for less bodily integrity. Aggregating multidimensional capabilities into one number imposes substitution rates the framework itself rejects (Robeyns 2017, Ch. 2). The applied-measurement literature has worked at this problem (Anand et al. 2009; Alkire & Foster 2011) but the tension does not go away.

Fourth, *functionings are achievements, capabilities are options*. Two people who have completed secondary school via a digital channel — same achievement, same functioning — may have arrived there through very different routes. One was free to choose; the other was coerced by the absence of any non-digital option. The capability claim is about what the person is *substantively free* to do, not only what they end up doing. This is why the capability approach is suspicious of pure-functioning indices: they record outcomes without asking whether the route was a free choice.

Fifth, the *list of capabilities* to attend to is not given. Nussbaum (2011) proposes a fixed list of ten central capabilities; Sen rejects fixity in favour of democratic deliberation; the applied literature compromises (Robeyns 2017). For our purposes the question is which capabilities matter for digital systems, and how one would know.

### 2.2 Adapting the approach to digital systems

Following Mansell (2017), Helsper (2021), Hilbert (2011), and Robinson et al. (2015), we propose a four-layer scheme for digital systems.

The *resources* layer asks what infrastructure, devices, services, and data flows are present: coverage, penetration, station counts, points of service. The *conversion factors* layer asks what allows a person to convert that presence into use: skills, language, gender norms, disability, affordability, time, trust. The *capabilities* layer asks what a person is substantively free to do or be online: use a service of their choice, transact safely, find work, exercise voice, refuse without penalty. The *functionings* layer asks what people actually do or achieve: bank account is used, child finishes online schooling, health appointment is kept, civic complaint is filed.

The four layers are not weights to be averaged; they are evidential strata. A capability index requires *every* domain to clear the capability layer at minimum, which is to say: every domain must have at least one indicator that registers a substantive freedom, not only a stock or a behaviour. A resource indicator can be informative — many are — but it cannot stand for a capability. The standard family fails this test routinely; we document it in §3.

### 2.3 The political economy of indicators

Indicators are governance technologies. Merry (2016) traces the rise of quantified indicators — from the Human Development Index through the Sustainable Development Goals — and shows how indicator design encodes a politics of legibility and intervention. Bhuta, Malito and Umbach (2018) extend the analysis across global-governance domains, documenting the capacity of an indicator, once instituted, to shape what becomes thinkable and fundable as policy. The Worldwide Governance Indicators are a paradigm case (Kaufmann, Kraay & Mastruzzi 2010): a composite produced for one analytic purpose has come to do regulatory and lending work it was not designed to do, and the contestation in the political-science literature (Apaza 2009; Thomas 2010; Langbein & Knack 2010) records the costs.

Reusing such an indicator in an index that promises something else is a particular failure mode. Bhuta et al. call it *indicator laundering*: borrowing the legitimacy of one composite to support a claim it was not built to support. The current DCLO build does this when it operationalises *Agency, Safety, Rights* through WGI percentiles (`03_indicator_validity_audit.md`, §1.4). The paper has to name this rather than absorb it.

### 2.4 The DPI-critique tradition

A specific literature has developed around India's digital public infrastructure. Khera (2022) and Rao (2019) document Aadhaar's exclusionary failures: ration shops that refused to dispense to citizens whose biometrics failed; pensioners denied entitlements through authentication errors; the displacement of older paper-based grievance mechanisms with digital ones that bypass the people who most needed them. Masiero (2023) places DPI in the *data-as-development* paradigm and asks what kind of development is being articulated when digital identification, payment rails, and consent layers become the precondition for social-protection access. Krishna (2024) examines the political economy of UPI saturation: who benefits from the volume, who absorbs the risk, who pays for the rails.

This literature is now substantial enough that an index-builder cannot scaffold a "Service Enablement" domain over Indian DPI without engaging it. To do so silently is to take a side. The framing here is therefore explicit: DCLO is offered as a critical instrument, not a marketing instrument. It is built to surface where DPI rhetoric outruns DPI capability; where the *existence* of UPI rails or eKYC service points has been read as the *substantive freedom* of citizens to use them safely, with consent, and with redress. The critique tradition shows the freedom does not follow automatically. The index has to be read with that knowledge or not at all.

### 2.5 Datafication and the costs of connection

Couldry & Mejias (2019, 2024) and Zuboff (2019) frame contemporary digital infrastructure as extractive — *data colonialism* and *surveillance capitalism* respectively. Their arguments are contested but they exert pressure on a particular kind of digital-capability measurement: an index that scores a country highly because its citizens transact frequently through digital channels may, on the data-colonialism reading, be scoring the depth of data extraction rather than the depth of capability. The capability approach itself does not commit to either framing; one can hold a Senian view of digital capability without subscribing to either Couldry-Mejias or Zuboff. But a capability-grounded measurement project must be aware that its functioning indicators — banking through a digital channel, completing schooling through a digital channel, accessing a state benefit through a digital channel — are not framing-neutral.

The implication is operational. The next-release outcome set (Findex digital-payment use, EGDI service-delivery, WHO digital-health uptake) is a functioning set, but its functioning content can be coerced rather than freely chosen. The capability layer is therefore not optional for the predictor: without indicators that register substantive freedom — gendered safe access, refusal capacity, voice and redress — a high functioning score cannot be distinguished from a high-uptake-under-coercion score. This is precisely the situation Khera (2022) documents in the Aadhaar case, and one that an index without a capability layer reads as success.

### 2.6 Reflexivity

Suchman (2007) and Eubanks (2018) press measurement projects — particularly those that are scaled, standardised, and consequential — to reflect on whose questions the measurement answers, whose silences it preserves, and whose harms it normalises. For DCLO the questions are concrete. *Whose questions?* Predominantly central-government and donor questions ("how digitally ready is the country?"). Less so frontline-user questions ("can I trust this app with my medical record?"). *Whose silence?* Indicators do not register people without official identity, women whose accounts are male-controlled, gig workers whose digital participation is forced. *Whose harm?* A poorly ranked country may be more, not less, attentive to dignity in service design; a high rank can normalise coerced inclusion.

These are not afterthoughts to a methodological paper. They are why the paper has to frame DCLO as *partial visibility* rather than verdict, and why the dashboard surfaces caveats alongside ranks. We position the index, after Mansell (2017) and Couldry & Mejias (2019), as a critical instrument: a device for diagnosing what the standard family registers and what it leaves out, not a device for ordering countries.

## 3. Construct architecture and indicator audit

### 3.1 Six domains, redefined as capability sets

The current build organises indicators into six domains: *Access and Connectivity (ACC)*, *Skills and Literacy (SKL)*, *Service Enablement (SRV)*, *Agency, Safety, Rights (AGR)*, *Economic Participation (ECO)*, and *Outcome Realisation (OUT)*. Under the capability framing each is redefined as a capability set — a bundle of substantive freedoms — and matched explicitly to layers in the four-layer scheme.

| Domain | Capability set (Sen-shaped) | Layer mix |
|---|---|---|
| ACC — Connectivity | The freedom to be reachable and to reach others through digital networks at affordable cost and without coerced exposure. | Resources + conversion factors. |
| SKL — Digital literacy and competence | The freedom to use digital tools with effective competence for personally chosen ends. | Conversion factors + capabilities. |
| SRV — Access to essential services through digital channels | The freedom to obtain health, education, finance, and government services through digital channels of one's choice, at no cost to dignity. | Resources + capabilities + functionings. |
| AGR — Agency, safety, rights | The freedom to act online without fear; to consent or refuse; to be free of surveillance and platform coercion; to exercise voice. | Capabilities. |
| ECO — Economic participation | The freedom to participate in the digital economy on terms that do not require accepting precarity, surveillance, or wage theft. | Conversion factors + capabilities. |
| OUT — Realised functionings | The achieved functionings — banked, employed, educated, healthy, civically participating — *via* digital channels. | Functionings. |

Two consequences follow immediately. First, OUT in the capability frame is a *functionings* domain, not an "outcome" domain in the macroeconomic sense. The current build's use of population growth and agricultural land share under OUT is therefore not a weak operationalisation of the capability claim but a category error against it. Second, AGR and ECO are not residuals; they are the heart of the capability claim, because that is where the freedom-versus-coercion distinction actually bites. A digital-capability index whose AGR domain is built from Worldwide Governance Indicators is an index that has decided not to measure agency.

### 3.2 The current build, indicator by indicator

The country-track build draws its inputs from `data/gold/dpi_selected_indicators_by_domain.json`. The full audit (`03_indicator_validity_audit.md`) classifies every indicator on three criteria: construct validity (does it target the layer the domain claims to measure?), measurement validity (does it measure the thing it operationally measures?), and panel coverage (how many of the 470 entity-year cells does it actually populate?). Of 20 unique indicators selected, **16 fail construct validity** under the four-layer scheme.

The category errors are explicit and concentrated.

*ACC* uses World Bank FDI inflows (BX.KLT.DINV.WD.GD.ZS), which is a capital-account indicator with no plausible mechanism to digital connectivity. ACC keeps secure-server and internet-user series with construct caveats but the FDI series is a hard fail.

*SKL* uses tertiary and secondary gross enrolment ratios (SE.TER.ENRR, SE.SEC.ENRR). These are general-education indicators. SDG 4.4.1 — proportion of youth and adults with at least basic ICT skills — is the canonical comparable cross-country indicator, maintained by ITU and UNESCO, and is straightforwardly available. The build does not use it. The original `docs/dclo-indicator-mapping.md` specifies that SKL_1 should be "digital literacy rate (or proxy via ICT training participation)"; the country-track operationalisation departs from its own specification.

*SRV* uses four WTO services-export proxies: total services exports, transport exports, travel exports, and postal-courier exports. These are macroeconomic services-trade indicators. They do not register whether households can access health, education, finance, or government through a digital channel; they register whether the country exports services. The construct error is not subtle: a country can export large volumes of business services while its citizens face constrained access to digital health and finance, and a country can have minimal services-trade and effective digital-service delivery. These are the indicators on which the headline causal result rests.

*AGR* uses Worldwide Governance Indicators — Control of Corruption percentile (CC.PER.RNK) and Political Stability percentile (PV.PER.RNK). These are expert-coded perception composites at country level (Kaufmann, Kraay & Mastruzzi 2010), contested in the political-science literature (Apaza 2009; Thomas 2010; Langbein & Knack 2010). They measure macro-political institutional quality, not digital agency. Reducing *digital* agency — consent, voice, complaint redress, gendered safe use, identity self-determination — to a percentile rank produced by another consortium for a different purpose is the indicator-laundering Merry (2016) and Bhuta et al. (2018) describe.

*ECO* uses Labour Force Participation Rate (SL.TLF.CACT.ZS), Unemployment (SL.UEM.TOTL.ZS), Trade as percentage of GDP (NE.TRD.GNFS.ZS), and GDP-per-capita growth (NY.GDP.PCAP.KD.ZG). None of these are digital-economy specific. They are macroeconomic structure indicators that should serve as controls, not as measures of digital economic participation.

*OUT* uses population growth (SP.POP.GROW), agricultural land share (AG.LND.AGRI.ZS), and access to clean cooking fuels (EG.CFT.ACCS.ZS). The first two have no plausible causal pathway from digital capability and are repurposed, in the revised identification strategy, as *placebo outcomes*. The third is weakly tied to digital capability and is demoted to a context covariate.

Three indicators have additional measurement-validity concerns. WB_SE.ADT.LITR.ZS (adult literacy) appears in some pre-aggregation views with **98.7 % missingness** (n = 7 observations) and is excluded from headline runs. WB_IT.NET.USER.ZS and WB_IT.CEL.SETS.P2 sit at **91.5 %** missingness in the full panel; they enter the gold table because the intake gate uses `pct_observed_global` (computed on populated rows) rather than panel-cell coverage. Effective panel coverage is in the **8 % range**, while the intake mapping rule (`docs/dclo-indicator-mapping.md`) specifies a 30 % missingness ceiling. The gating is not enforcing the rule it claims.

### 3.3 Layer mismatch — a summary

Read against the four-layer scheme, the current country-track build mostly populates the *resources* layer with a handful of weakly-related conversion-factor indicators sprinkled in. The *capabilities* layer is essentially empty. The *functionings* layer is operationalised by macroeconomic stocks that are not functionings in any capability sense. The composite is therefore a resource-and-readiness index dressed in capability vocabulary. This is a fixable problem; it is not a small one.

The recommended replacement set, drawn in detail in `03_indicator_validity_audit.md`, draws from sources the index-making community already uses: ITU's Digital Development Dashboard (mobile-broadband subscriptions, 4G+ coverage, price-of-1GB-as-percent-GNI, SDG 4.4.1 ICT skills); the World Bank's Findex (digital-payment use, account use, female-controlled accounts); the World Bank's GovTech Maturity Index; the UN's E-Government Development Index; GSMA's Mobile Internet Connectivity Index and Mobile Gender Gap; Freedom House's Freedom on the Net (Obstacles to Access, Limits on Content, Violations of User Rights); Privacy International's data-protection laws status; ILOSTAT's digitally-mediated employment shares; the Online Labour Index; UNESCO's digital-education uptake; WHO's digital-health indicators; and SDG 16.10.2 (right-to-information statutory frameworks). For the India track the replacement set draws on NSS-PLFS, NFHS, NCRB, NPCI's UPI dashboard, UIDAI authentication transactions, eSanjeevani teleconsults, MoHFW PMJAY claims, NHA, and ASER's digital-skills supplement when it is published.

### 3.4 The state-track: two cross-sections, not a panel

The India state-year track contains 95 rows. The years populated are dominated by NFHS-4 (field-work 2015–16) and NFHS-5 (field-work 2019–21); the binding constraint is the NFHS field cycle. The dashboard markets this as a "panel". The honest description is **two cross-sections**, with limited supplementary material from RBI DBIE, MoSPI, and NPCI national aggregates that, on inspection, are *constants across states within a year* — `NAT_upi_total_vol`, `NAT_upi_total_val`, and several internet-banking series enter the state table identically for every state-year. They contribute nothing to within-year between-state variation.

The paper's commitment is to reframe the state track as a *repeated cross-section* in this draft and to roadmap a path to an annual panel, with new state-disaggregated indicators ingested from TRAI Performance Indicator Reports (rural/urban internet-user shares by state), NPCI state dashboards (UPI per-capita), CSC e-Governance Services (CSC operations by state), UIDAI dashboards (state-wise authentication), MoHFW (state-wise eSanjeevani teleconsults), NeGD (state-wise DigiLocker registrations), PFMS (state-wise DBT transfers), eNAM (state-wise trade volume), NHA (state-wise PMJAY digital-claims share), CBIC (state-wise GST registrations), NCRB (cyber-fraud reports per million), and DoPT (state-wise RTI online filings per capita). All of these are public.

### 3.5 The state-track context-adjustment is constant across states

The dashboard exposes a `DCLO_score_context_adjusted` toggle that purports to adjust state DCLO scores by India's national DPI context. Inspection of the gold state-year table shows `CTX_dpi_composite_v2` and the corresponding z-score are identical for every row. The "context-adjusted" series is therefore a rigid additive shift — a global level adjustment with zero comparative information — and is currently misnamed.

### 3.6 The construct-validity gates the next release will enforce

The acceptance criteria for the next release, formalised in `03_indicator_validity_audit.md` §4 and to be locked at pre-registration, require each indicator to clear a layer assignment with a documented mechanism, panel-cell coverage of at least 0.40, cross-source triangulation where possible, and disaggregability by gender or by urban-rural where the source permits. Failures on (1)–(4) exclude the indicator; failures on the disaggregability and stability criteria flag it. The current build clears none of these gates as written.

## 4. Empirical comparison with the standard family

A capability-grounded reading of the standard family is best surfaced by comparison. Where DCLO and a sibling index agree, the agreement is mostly the residue of shared inputs. Where they disagree, the disagreement is informative — and the informativeness is, for our purposes, the point.

### 4.1 The comparator universe

We propose Spearman rank correlations between DCLO and four sibling indices, restricted to the panel-overlap subset and to the years for which both are available: the ITU ICT Development Index (relaunched edition); the WEF Network Readiness Index; the World Bank GovTech Maturity Index; and the UN E-Government Development Index. The pairwise correlations cannot be reported in this draft — the ingestion of comparator series is part of the pre-registered next-release work — but the comparison design is described here so that the reading frame is fixed before the numbers are computed.

We expect three patterns to emerge. First, all pairwise correlations should be positive and, in most years, large. The shared inputs ensure that. Second, the correlations should fall when restricted to the *capability layer* of DCLO (the subset of indicators that target capabilities or functionings rather than resources), because the standard family populates the capability layer thinly. Third, the correlation should fall further when DCLO is computed under the construct-validated indicator set documented in `03_indicator_validity_audit.md`, because the substitutions push DCLO to register what the standard family does not. The pre-registration locks these as predictions.

### 4.2 Where disagreement is diagnostic

We illustrate the diagnostic logic with three stylised cases, using the placeholder logic of the comparison rather than the numbers (which await ingestion). The cases are drawn from the patterns one expects from the audit.

*Case A: a country highly ranked by EGDI but lower by DCLO because AGR fails.* Several Gulf states are well known to score highly on e-government readiness — service-portal density, telecommunication infrastructure, online-service availability — while scoring poorly on press freedom, civic-space, and digital-rights indicators tracked by Freedom House Freedom on the Net and Article 19. Under DCLO with the AGR replacement set the rank would fall substantially. The disagreement is diagnostic: a service portal whose use is conditioned on the absence of meaningful redress is not a substantive freedom. The standard family does not register this; DCLO redefined does.

*Case B: a country highly ranked by GovTech but lower by DCLO because the functioning layer is missing.* Several middle-income countries have invested heavily in the GovTech Maturity Index's pillars — Core Government Systems, Public Service Delivery, Citizen Engagement, GovTech Enabler — without commensurate uptake by households. Findex and EGDI's complaint-resolution sub-indicator, where they exist, would surface low actual usage. DCLO under the replacement set would register the gap; GovTech, by design, does not.

*Case C: a country highly ranked by ITU IDI but lower by DCLO because conversion factors fail.* Several South Asian and Southeast Asian states have rapidly closed the connectivity gap (mobile-broadband subscriptions, 4G+ coverage) while leaving conversion factors — digital skills, gendered access, language and disability accommodation — substantially open. ITU IDI's resource-heavy structure rewards the connectivity build-out; DCLO with conversion-factor indicators (GSMA Mobile Gender Gap; price-of-1GB-as-percent-GNI; SDG 4.4.1) would not.

The diagnostic logic is consistent across the three cases. DCLO is not "right" and the comparators "wrong"; DCLO is *differently right* in a way that exposes what the standard family does not measure. The next release will report the rank-correlation matrix and the country-level disagreements with the appropriate caution about panel coverage.

### 4.3 What disagreement is *not*

Two readings of the disagreement we want to foreclose. First, the disagreement is *not* a methodological failure of the comparators — they are doing what they were built to do. Second, the disagreement is *not* a policy verdict — the index's job is to make the disagreement visible, not to adjudicate. The reflexive caveat in §7 below reapplies here. The instrument's contribution is in the diagnostic, not in the ordering.

## 5. Identification: what the headline TWFE result actually says

This section is the methodological centre of the paper. It does the work of showing that the published causal-evidence layer of DCLO, taken at face value, is largely a structural-overlap artefact, and that the honest reading is therefore a diagnostic one rather than a "DCLO causes service outcomes" reading.

### 5.1 The result on the page

The current build's headline causal result is a two-way fixed-effects estimate of `SRV_score_t = β · DCLO_score_lag1 + γ · model_trust_tier_lag1 + ν_i + λ_t + ε_{i,t}`, with cluster-robust standard errors at the entity level. From `data/gold/dclo_causal_coefficients.csv`:

- *Baseline TWFE (l1):* β = 0.624 on `DCLO_score_lag1`, SE = 0.100, t = 6.25, p ≈ 4.2 × 10⁻¹⁰, 95 % CI [0.428, 0.820], n = 470, R²_within ≈ 0.381.
- *Robustness TWFE (l2):* β = 0.319 on `DCLO_score_lag2`, SE = 0.094, p ≈ 7.3 × 10⁻⁴, 95 % CI [0.134, 0.504]. The two-period lag attenuates by roughly half.
- *Robustness TWFE (l1, no controls):* β = 0.465, SE = 0.121, p ≈ 1.3 × 10⁻⁴.
- *Pooled OLS (l1):* β = **0.998**, SE = 0.025, t = 39.5, 95 % CI [0.949, 1.048], R² = 0.910.
- *Placebo (within-year permutation):* β = −0.014, SE = 0.009, p ≈ 0.119.

Read as written, the baseline result looks substantial: a one-unit shift in the lagged composite is associated with a 0.62-unit shift in next-period service-enablement, the placebo is non-significant, and the lag-2 specification yields a smaller-but-still-positive coefficient. A reviewer reading only the dashboard would record a positive within-entity association robust to placebo permutation.

### 5.2 The pooled-vs-FE gap is a structural-overlap signature

The pooled OLS coefficient β ≈ 1.00 is the giveaway. Pooled-OLS regressing a composite on its own one-period lag, when the composite and the outcome are built largely from the same indicator inputs, is mechanically near-unity. The R² of 0.91 reinforces the reading: nearly all the cross-sectional variation in `SRV_score_t` is explained by `DCLO_score_lag1`, and that explanatory power is being delivered by the structural overlap between predictor and outcome rather than by any behavioural channel. SRV is constructed from four WTO services-trade exports (total, transport, travel, postal-courier); DCLO is constructed as the equally weighted mean of available domain z-scores including SRV; therefore `DCLO_lag1` and `SRV_t` share a substantial fraction of their variation by construction.

The within-entity demeaning of the TWFE specification reduces this overlap by sweeping out time-invariant country structure, but it does not eliminate it. The residual β ≈ 0.62 still carries a significant share of the structural-overlap content: the year-on-year variation in WTO services exports is correlated with the year-on-year variation in the DCLO composite that includes those exports as one of its four SRV indicators. The within-FE coefficient is not zero — there is some additional within-country signal — but the read is that it is a contaminated estimate, not a clean one. Calling β = 0.62 "the causal effect of digital capability on service enablement" inverts the diagnostic. The diagnostic is that the construct is over-determined by indicator overlap, which is exactly the failure the audit flagged.

### 5.3 The mechanism of the contamination

It is worth being precise about why the contamination operates the way it does. The composite *DCLO_score_t* is the equally weighted mean of available domain z-scores at *t*, including the SRV domain. SRV at *t* is itself a z-score average over the four WTO services-trade exports for that country-year. The outcome variable in the headline regression is *SRV_score_t*. Therefore *DCLO_score_{t-1}* shares with *SRV_score_t* (i) the *level* component — the country's persistent services-trade structure, which TWFE sweeps out via entity FE — and (ii) the *trend* component — the global services-trade movement that affects most countries similarly, which year FE largely sweeps out — but also (iii) a *country-by-year* component arising from the same indicator-construction route generating both predictor and outcome with one period's lag.

Component (iii) is the residual that survives within-FE demeaning. In a panel where year-on-year movements in the four WTO indicators are positively autocorrelated (which they are, because services-trade is sticky), the lag-1 association between DCLO and SRV is mechanically positive. The pooled OLS estimate of β ≈ 1.00 with R² ≈ 0.91 is the unbounded version of this; the within-FE estimate of β ≈ 0.62 is the bounded version, with the bounds set by entity demeaning. The lag-2 estimate of β ≈ 0.32 attenuates further because the autocorrelation in WTO services-trade weakens with lag length. None of these movements registers a behavioural channel from "digital capability" to "service enablement"; they register the indicator-construction route.

This reading does not require deep econometric machinery. It requires reading the construction of predictor and outcome side by side. The reason the audit (`03_indicator_validity_audit.md`) is the load-bearing component of the paper, and not the regression table, is that the audit is what shows the contamination exists. The regression table merely registers it.

### 5.4 The method-comparison flip

The single most diagnostic finding in the build is in `data/gold/dclo_method_comparison.csv`. It reports, for each year 2015–2024, the Spearman rank correlation between the *baseline* DCLO ranking, a *confidence-weighted* DCLO ranking (which weights domain z-scores by the model-trust tier), and a *causal-signal* ranking (the within-panel residualised score). The key column is the baseline-vs-causal correlation across years:

| Year | ρ (baseline vs causal) |
|---|---|
| 2015 | −0.190 |
| 2016 | **−0.547** |
| 2017 | −0.416 |
| 2018 | −0.170 |
| 2019 | +0.292 |
| 2020 | +0.379 |
| 2021 | −0.427 |
| 2022 | +0.209 |
| 2023 | +0.533 |
| 2024 | **+0.759** |

Baseline-vs-weighted correlations are uniformly above 0.96 for all ten years; the two methods that share most of their construction agree, as one would expect. The baseline-vs-causal correlation, however, swings from a strong negative (−0.55 in 2016) to a strong positive (+0.76 in 2024), with substantial year-on-year noise in between. The dashboard previously presented this series as routine "method agreement over time"; that framing was wrong.

Interpreted under the structural-overlap reading of §5.2, the flip is what one would expect from a panel where the two construction routes — straight aggregation and within-panel causal residualisation — share most of their indicator content but partition it differently across the years as the country-year cells shift. The flip is not registering a real change in the underlying capability landscape; it is registering the construction's instability under reasonable methodological perturbation. The diagnostic is severe. A construct that yields ρ = −0.55 in one year and ρ = +0.76 eight years later under a within-panel re-weighting is not measuring a stable underlying quantity. It is measuring a mixture of the indicator inputs and the panel composition.

We thus have two diagnostic findings, not one. The pooled-vs-FE gap shows that the TWFE coefficient is largely a structural-overlap artefact in cross-section. The method-comparison flip shows that the same construct is unstable under within-panel re-weighting across years. Together they make the case that the published "causal-evidence" layer is a reading of the indicator construction as much as it is a reading of any underlying digital-capability process.

### 5.5 The placebo passes too easily

The published placebo is a within-year permutation of `DCLO_score`: re-shuffle the 47 entities' DCLO scores within each year, re-fit the TWFE, and check that β attenuates. The result (β = −0.014, p ≈ 0.119) is reported as evidence of falsification. It is a weak falsification.

A within-year permutation breaks the cross-sectional structure entirely. Any signal that depends on the cross-sectional ordering — including spurious signals — will attenuate under within-year permutation. The expected result of this placebo, under the structural-overlap reading, is exactly the result observed. It does not distinguish a capability claim from an indicator-overlap claim. It is uninformative.

A more disciplined falsification battery, set out in `05_robustness_protocol.md`, includes: a *placebo outcome* — a variable that should not respond, such as agricultural land share (which appears in the build under OUT) or annual rainfall — to establish that the within-FE association is specific; a *leave-future-out* placebo that regresses future DCLO on present outcome to test the directionality assumption; an *event-study* placebo around DPI rollouts (UPI 2016, Aadhaar saturation 2014–2018, India Stack 2017–2019, PIX 2020) to test parallel pre-trends; a *within-country* permutation to complement the within-year version; and a *pre-treatment-window* test that estimates the baseline specification on data restricted to years before each major DPI rollout. The next release implements P1–P6 in the protocol.

### 5.6 What the result says, then

The honest read of the current causal layer, taken at face value, is that the TWFE coefficient β ≈ 0.62 on lagged DCLO is a *within-entity, within-period conditional association between a macro-structural composite measured at t–1 and a closely related macro-structural composite measured at t, under explicit unconfoundedness assumptions, contaminated by the structural overlap of predictor and outcome*. It is consistent with the structural-overlap explanation; it is not strong evidence for a behavioural digital-capability channel. Under the construct-disjoint redefinition pre-registered for the next release — outcome built only from functioning-layer indicators (Findex digital-payment use, EGDI service-delivery, WHO digital-health uptake), predictor built only from resource, conversion, and capability layers, with no shared source — the expected association attenuates substantially. The pre-registration commits to reporting whatever the new estimate is, including null and negative results.

The methodological centre of the paper is therefore not the coefficient. It is the demonstration that the coefficient as published is a reading of the construction. That demonstration is the contribution. It is also why the paper's contribution is critical and methodological rather than rank-substitutive. We do not propose a new headline number. We propose a way to read the existing one without being misled by it.

## 6. A revised research design

If the paper's reading of the current causal layer is to be more than a critique, it has to specify what would replace it. This section sets out the revised research design — the estimand, the DAG, the robustness battery, and the pre-registration commitment. The full specifications are in `04_identification_strategy_revised.md` and `05_robustness_protocol.md`.

### 6.1 The estimand

Let *i* index countries (or Indian states), *t* index years, *Y_{i,t}* a *capability-targeted* outcome (a functioning, e.g., share of households that completed a financial transaction through a digital channel in the past year), and *D_{i,t}* a *capability-targeted* digital-capability composite that does not share constituent indicators with *Y*. The estimand is

> β = E [ Y_{i,t} | D_{i,t-1} = d+1, X_{i,t}, ν_i, λ_t ] − E [ Y_{i,t} | D_{i,t-1} = d, X_{i,t}, ν_i, λ_t ]

i.e. the change in the expected functioning level associated with a one-unit change in lagged capability, conditional on time-varying controls *X_{i,t}* (GDP per capita, urbanisation, demographic structure, electricity access, public-spending share), entity fixed effects ν_i, and year fixed effects λ_t. Under the unconfoundedness assumption that, conditional on *(X, ν, λ)*, no further confounder varies at *i × t* and affects both *D_{t-1}* and *Y_t*, β recovers an average treatment effect on the treated. Under any weaker assumption β is reported as a within-entity conditional association. The current build's estimand is mis-specified because *Y* and *D* share constituent indicators; the proposed estimand becomes correct only after the indicator-validity audit is implemented.

### 6.2 The DAG

The DAG specifies four backdoor channels from *D_{t-1}* to *Y_t*: time-invariant entity structure ν_i, closed by entity FE; global shocks λ_t, closed by year FE; time-varying confounders (GDP per capita, urbanisation, demographic structure), closed by *X*; and country-year-specific reform shocks *R_{i,t}* (UPI 2016, India Stack 2017–2019, PIX 2020, pandemic stay-home 2020), which TWFE *cannot* close. The binding identification threat is the reform-shock channel, because reforms generally drive *D* and *Y* near-simultaneously.

### 6.3 The robustness battery

The pre-registered battery is fifteen specifications and six placebos. The specifications include the baseline TWFE (S1) and lag-length variant (S2); region-by-year fixed effects (S3) for sub-global shocks; control-sensitivity (S4); pooled OLS (S5) and first-difference (S6) as benchmarks; Arellano–Bond GMM (S7) for the lagged-DV concern; an errors-in-variables IV (S8) instrumenting the construct-validated DCLO with a *disjoint-source* sibling composite (e.g., D-A from ITU + Findex, D-B from GSMA + GovTech); long-difference (S9) for structural change; three synthetic-control case studies (S10 India around UPI 2016; S11 Estonia around e-residency 2014; S12 Brazil around PIX 2020) following Abadie & L'Hour (2021); a dynamic event-study DiD (S13); spatial-lag controls (S14) for SUTVA; and a population-weighted re-estimation (S15) for selection-on-coverage.

The placebo set extends beyond within-year permutation: P1 within-year, P2 within-country, P3 placebo outcome on agricultural land share, P4 placebo outcome on rainfall, P5 leave-future-out reverse-direction, P6 pre-treatment-window parallel-trends.

The stability set adds Dirichlet weight perturbation (St1), leave-one-domain-out (St2), leave-one-country-out (St3), leave-one-year-out (St4), cluster bootstrap (St5), Manski bounds under worst-case unobserved-indicator values (St6), and Cinelli–Hazlett (2020) sensitivity for minimum unobserved-confounder strength to overturn the headline (St7).

### 6.4 The decision rule

A finding is reported as causal-evidence-supported only if the coefficient retains its sign across S1, S2, S3, S6, S8, and S15; the placebos P1, P2, P3, P5, P6 return coefficients smaller in magnitude than the baseline; the Cinelli–Hazlett robustness value RV is at least 0.5 of any one observed control's strength; and the leave-one-out distribution from St3 contains zero in fewer than 5 % of draws. If any of these fail, the finding is reported as a *conditional association*, not a causal effect, with the failure foregrounded.

### 6.5 Errors-in-variables and the disjoint-source IV

The errors-in-variables specification (S8) deserves a sentence on its own because it is the cleanest mitigation of the construct-contamination problem the paper centres on. The idea is to construct two composites of the same construct from disjoint sources. *D-A* is built from ITU connectivity series, Findex digital-payment use, GovTech CGSI, and SDG 4.4.1 ICT skills. *D-B* is built from GSMA Mobile Internet Connectivity, Freedom House Freedom on the Net, EGDI Online Services, and ILOSTAT digitally-mediated employment. The two composites share no source. Under classical measurement-error assumptions, *D-B* is a valid instrument for *D-A* (it is correlated with *D-A* through the underlying construct and uncorrelated with the regression residual through source partition), and the 2SLS estimate provides a measurement-error-corrected coefficient. Hjort & Poulsen (2019) use a related geographic-luck instrument (submarine cable arrival) for the resource layer; the disjoint-source IV is admissible at the construct layer. The pre-registration commits to reporting both 2SLS and the first-stage F.

### 6.6 The pre-registration commitment

The analysis plan is locked at `11_preregistration.md` before the next data refresh, with all specifications, placebos, stability tests, decision rules, and reporting formats committed in advance. The commitment is to report the result regardless of direction or significance. The construct-disjoint redefinition is the centerpiece: outcome from Findex digital-payment use, EGDI service-delivery, WHO digital-health uptake; predictor from ITU IDI annual indicators, GSMA Mobile Internet Connectivity Index, GovTech Maturity Index, Freedom House FOTN, ILOSTAT digitally-mediated employment, SDG 4.4.1 ICT skills; with no shared source between predictor and outcome.

The expected outcome under the structural-overlap reading of §5 is substantial attenuation of the headline coefficient — possibly to a null. The pre-registration's most important property is that this null, if it appears, is reportable as a finding rather than as a failure.

## 7. Ethics, reflexivity, and the politics of measurement

A digital-capability index that ranks countries is a governance technology in Sally Engle Merry's sense. The full ethics statement is in `06_ethics_and_responsible_use.md`; this section names the politics directly.

*Indicator laundering.* Reusing indicators built for one purpose in an index whose name promises something else transfers the legitimacy of the source onto a claim the source cannot support. The current build does this when it operationalises *Agency* as Worldwide Governance Indicators and *Service Enablement* as WTO services-trade. The fix is not to layer caveats on top of the same indicator set; the fix is to drop the indicators and replace them. This paper commits the next release to that replacement.

*Rendering invisible.* The country panel includes 47 economies, most of them upper-middle and high-income. Several South Asian peers, most low-income African states, and small island developing states are absent. The panel composition follows the World Bank's DPI-comparator panel; the exclusions are not marked. Saying "the comparative track" without naming this exclusion is a category error. Every release publishes an inclusion list and an exclusion list with reasons; the paper foregrounds the panel limit explicitly.

*Aggregation disguising distributional harm.* A composite score is a population mean of indicators that are themselves means or shares. If a country has a high mean digital capability and a long tail of digitally-excluded households, the mean rewards the country. Capability theory rejects aggregation that masks inequality (Robeyns 2017, Ch. 2). Helsper (2021) and Hilbert (2011) show digital divides operate at multiple levels and across multiple cleavages — gender, urban-rural, class, age. None of these decompositions is in the current build's aggregate. Every release that uses an aggregate from the next refresh onward also reports either a gender decomposition, an urban-rural decomposition, or an Atkinson-style inequality penalty for at least three domains, with the absence named where data do not yet permit.

*The index as governance technology.* High DCLO scores can be used to legitimate further DPI investment that, locally, is exclusionary. Low scores can be used to legitimate top-down DPI imposition framed as catching up. Regional comparators can be invoked rhetorically in policy debate without the underlying construct supporting the rhetoric. None of this is hypothetical; the literature on Aadhaar and UPI documents the patterns (Khera 2022; Masiero 2023; Krishna 2024; Rao 2019). The paper acknowledges these uses and the dashboard surfaces them as caveats. They do not disappear because the index-builder is well-intentioned.

*Standpoint.* The author is a PhD researcher at JSGP, OPJGU, supervised by Prof. Swagato Sarkar. The author has no commercial funding, no consulting role with any government or platform, and no financial interest in any DPI vendor. Sarkar's lineage on the political economy of infrastructure and labour shapes the framing here, and the paper engages with the JSGP critical tradition explicitly. The standpoint is named so the reader can weight it.

A measurement instrument that enters a politicised field carries responsibilities a methodological paper alone cannot fully discharge. The minimum is to name the politics, to mark the silences, to refuse the rhetoric of "objective" rankings, and to keep the construct's failures visible. We hold to that minimum here and commit to extending it in the next release.

## 8. Discussion

The paper claims a methodological and critical contribution. It does not claim a new ranking, a verdict on which countries are doing DPI well, or a policy recommendation that follows from the index. The instrument is the diagnostic; the diagnostic is the contribution.

Three concrete uses follow.

*First, flagging where DPI-success narratives outrun capability evidence.* When India's DPI rhetoric, or any country's, points to indicator-family rankings as evidence of capability gain, the diagnostic offers a corrective. The standard family does not register what the rhetoric claims; it registers what the build-out has done. The two are not the same. The instrument names the gap and forces the rhetoric to either address it or to drop the capability vocabulary.

*Second, directing the next-generation indicator build.* The replacement set in §3.3 is not aspirational — every indicator in it is publicly available and instituted by an ongoing data-producing organisation. SDG 4.4.1 is maintained by ITU and UNESCO. Findex is on a fourth wave. GovTech Maturity has expanded to 198 economies. EGDI is biennial and stable. Freedom House Freedom on the Net is annual. GSMA's gendered access work is in its tenth year. ILOSTAT carries digitally-mediated employment shares. The DCLO 2026 release that the pre-registration locks will rebuild on this set. The instrument's diagnostic shows where the rebuild is necessary; the next release will demonstrate that it is feasible.

*Third, informing the JSGP research programme on digital infrastructure political economy.* The framework here connects to a longer research programme on the political economy of digital infrastructure — on how DPI investments redistribute risk, restructure formal-informal boundaries, and shape the conversion factors through which substantive freedom emerges. The instrument provides one analytic surface for that work: a way to read the indicator family with the critique tradition rather than against it. The dashboard is the artefact; the paper is the critique; the research programme is the larger frame the artefact and the critique are accountable to.

The reading of disagreements with sibling indices in §4 deserves a brief gloss. We did not present DCLO disagreement with EGDI, GovTech, ITU IDI, or NRI as a claim about which index is correct. None of the indices is correct in the sense the capability framing demands; all of them, including DCLO under its current build, fail capability validity for substantially the same reasons. The disagreement is a topology of the failure surface: it shows where the standard family's silences fall, and where the substitutions of the next-release indicator set would push the rank order. That is a more modest contribution than "DCLO is the better index", but it is also more defensible. A critical instrument is allowed to be modest about its own ranking; what it has to be precise about is which silences it surfaces.

The reading also has implications for the policy use of digital indicators. Multilateral and bilateral DPI investment is increasingly indexed to instruments of the kind §4 catalogues — country selection for World Bank GovTech operations, EU Global Gateway tranches, G20 DPI cooperation programmes, ITU country-engagement plans. The instrument-rank is a procurement gate. When the rank does not register what the rhetoric promises, the procurement gate filters on the wrong signal. Naming this is not anti-DPI; it is in the interest of DPI investment that lands well. The capability framing, applied properly, sharpens the rank rather than blunting it. The more the instrument registers conversion factors, the more reliably it picks countries where DPI investment will translate into substantive freedom for citizens rather than into stranded infrastructure or coerced inclusion.

The instrument's limits are well-defined. It does not measure intra-country inequality. It does not see microdata. It is bounded to the panel its inputs cover. It cannot adjudicate between competing policy regimes; it can only show what the standard family does and does not register. These limits are features, not bugs, of a critical instrument. A measurement device that admits its scope is more trustworthy than one that does not.

## 9. Limitations

*Construct overlap in the current build.* The indicator-validity audit identifies category errors in 16 of 20 country-track indicators. The headline TWFE coefficient is contaminated accordingly. The paper foregrounds this as the diagnostic finding but the coefficient must be read with the contamination in mind, not against it. The construct-disjoint redefinition is committed for the next release and is not yet implemented.

*State-track is two cross-sections.* The state-year track is 95 rows over the NFHS-4 and NFHS-5 cycles. It is not yet a panel. The paper reframes it as a repeated cross-section and roadmaps the annual-panel build; until that build is delivered, the state track supports descriptive comparison only, not within-state causal estimation.

*The 47-country panel excludes most of the Global South.* Several South Asian peers, most low-income African states, and small island developing states are absent. Inference does not generalise outside the included panel. Population-weighted re-estimation (S15) and explicit inclusion/exclusion tables in every release are mitigations but not solutions; the panel limit is structural to the indicator family.

*No microdata.* The build uses macro composites only. ITU IDI, ITU ICT-skills, IFC Findex micro file, OECD PIAAC, UNDP-OPHI MPI, NSS-PLFS, NFHS micro — none are ingested at micro level. The phase-2 plan in `12_known_issues.md` records this as priority work; the current paper makes no individual-level claim.

*Macro composites do not register intra-country inequality.* The aggregate score is a population-mean construction. Even after the pre-registered next release adds gender and urban-rural decomposition for at least three domains, the within-country distributional structure for the remaining domains is not surfaced. An Atkinson-style penalty is mooted; the data infrastructure to support it is partial.

*The paper's reliance on the existing dashboard build.* Many of the empirical claims here — the pooled-vs-FE gap, the method-comparison flip, the panel coverage statistics — derive from the build whose construct validity the paper questions. The reading is consistent: the build's diagnostic findings are about the build itself, not about an underlying capability process. The reader should weight this consistency rather than treat it as an inconsistency. The next release tests whether the diagnostic survives when the construct is corrected.

## 10. Conclusion

The dashboard is an instrument, not an oracle. The paper's contribution is to read the standard cross-national digital-development indicator family through the capability approach, to find specific places where the family fails capability validity, and to commit to a construct-validated next release that pre-registers its analysis plan in advance. The headline conclusions are three.

First, the family does not, on its own measurement-theoretic terms, measure digital capability in any sense the capability approach would recognise. It measures macro-economic stocks with a digital-access overlay. Calling it "digital capability" is a vocabulary choice that the construct cannot support without the substitutions documented here.

Second, the conventional within-entity TWFE result that lagged DCLO predicts service-enablement is largely a structural-overlap artefact. The pooled-OLS sibling at β ≈ 1.00 shows it; the method-comparison flip from −0.55 in 2016 to +0.76 in 2024 confirms it; the construct-validity audit explains it. The honest read is diagnostic: the contamination is the finding.

Third, the family can be repaired. The replacement indicator set is real, public, and ingestable. The construct-disjoint redefinition is specifiable, pre-registrable, and falsifiable. The pre-registration commits to reporting the result whether positive, null, or negative. The instrument's value is not in the current ranking but in the route from current ranking to repaired construct.

The dashboard publishes the build, the audit, the data, the code, and the pre-registration. The paper articulates the critique. The research programme — at JSGP, in conversation with the political economy of digital infrastructure — is the larger frame. We offer DCLO as a critical instrument, not a verdict. The standard family will continue to do the work it does well; the question this paper presses is what work it cannot do, and what would have to change before it can.

---

## References

Abadie, A., & L'Hour, J. (2021). A penalized synthetic control estimator. *Journal of the American Statistical Association*, 116(536), 1817–1834.

Alkire, S., & Foster, J. (2011). Counting and multidimensional poverty measurement. *Journal of Public Economics*, 95(7–8), 476–487.

Anand, P., Hunter, G., Carter, I., Dowding, K., Guala, F., & van Hees, M. (2009). The development of capability indicators. *Journal of Human Development and Capabilities*, 10(1), 125–152.

Anand, P., & van Hees, M. (2006). Capabilities and achievements: An empirical study. *Journal of Socio-Economics*, 35(2), 268–284.

Apaza, C. R. (2009). Measuring governance and corruption through the worldwide governance indicators: critiques, responses, and ongoing scholarly discussion. *PS: Political Science & Politics*, 42(1), 139–143.

Bhuta, N., Malito, D. V., & Umbach, G. (Eds.). (2018). *The Palgrave Handbook of Indicators in Global Governance*. Palgrave Macmillan.

Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.

Christensen, G., & Miguel, E. (2018). Transparency, reproducibility, and the credibility of economics research. *Journal of Economic Literature*, 56(3), 920–980.

Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable bias. *Journal of the Royal Statistical Society B*, 82(1), 39–67.

Cobbe, J., & Singh, J. (2024). Public-sector duties in indicator design (forthcoming).

Couldry, N., & Mejias, U. A. (2019). *The Costs of Connection: How Data Is Colonizing Human Life and Appropriating It for Capitalism*. Stanford University Press.

Diamantopoulos, A., & Winklhofer, H. M. (2001). Index construction with formative indicators: an alternative to scale development. *Journal of Marketing Research*, 38(2), 269–277.

Eubanks, V. (2018). *Automating Inequality: How High-Tech Tools Profile, Police, and Punish the Poor*. St. Martin's Press.

Hair, J. F., Sarstedt, M., Ringle, C. M., & Mena, J. A. (2012). An assessment of the use of partial least squares structural equation modeling in marketing research. *Journal of the Academy of Marketing Science*, 40, 414–433.

Helsper, E. J. (2021). *The Digital Disconnect: The Social Causes and Consequences of Digital Inequalities*. Sage.

Hilbert, M. (2011). The end justifies the definition: the manifold outlooks on the digital divide and their practical usefulness for policy-making. *Telecommunications Policy*, 35(8), 715–736.

Hjort, J., & Poulsen, J. (2019). The arrival of fast internet and employment in Africa. *American Economic Review*, 109(3), 1032–1079.

Jarvis, C. B., Mackenzie, S. B., & Podsakoff, P. M. (2003). A critical review of construct indicators and measurement model misspecification. *Journal of Consumer Research*, 30(2), 199–218.

Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators: methodology and analytical issues. World Bank Policy Research Working Paper 5430.

Khera, R. (2022). *Dissent on Aadhaar: Big Data Meets Big Brother*. Orient Black Swan.

Krishna, S. (2024). The political economy of digital public infrastructure. *The Information Society*, forthcoming.

Langbein, L., & Knack, S. (2010). The worldwide governance indicators: six, one, or none? *Journal of Development Studies*, 46(2), 350–370.

Manski, C. F. (1995). *Identification Problems in the Social Sciences*. Harvard University Press.

Mansell, R. (2017). *Imagining the Internet: Communication, Innovation, and Governance*. Oxford University Press.

Mansell, R., & Wehn, U. (Eds.). (1998). *Knowledge Societies: Information Technology for Sustainable Development*. Oxford University Press.

Masiero, S. (2023). Should we still be doing ICT for development research? *Information Systems Journal*, 33(5), 942–956.

Mejias, U. A., & Couldry, N. (2024). *Data Grab: The New Colonialism of Big Tech and How to Fight Back*. WH Allen.

Merry, S. E. (2016). *The Seductions of Quantification: Measuring Human Rights, Gender Violence, and Sex Trafficking*. University of Chicago Press.

Nosek, B. A., Alter, G., Banks, G. C., Borsboom, D., Bowman, S. D., Breckler, S. J., et al. (2015). Promoting an open research culture. *Science*, 348(6242), 1422–1425.

Nussbaum, M. C. (2000). *Women and Human Development: The Capabilities Approach*. Cambridge University Press.

Nussbaum, M. C. (2011). *Creating Capabilities: The Human Development Approach*. Harvard University Press.

OECD/JRC. (2008). *Handbook on Constructing Composite Indicators: Methodology and User Guide*. OECD Publishing.

Petter, S., Straub, D., & Rai, A. (2007). Specifying formative constructs in information systems research. *MIS Quarterly*, 31(4), 623–656.

Pradhan, P., Costa, L., Rybski, D., Lucht, W., & Kropp, J. P. (2017). A systematic study of Sustainable Development Goal (SDG) interactions. *Earth's Future*, 5(11), 1169–1179.

Rao, U. (2019). Population meets database: aligning personal, documentary and digital identity in Aadhaar-enabled India. *South Asia: Journal of South Asian Studies*, 42(3), 537–553.

Robeyns, I. (2017). *Wellbeing, Freedom and Social Justice: The Capability Approach Re-Examined*. Open Book Publishers.

Robinson, L., Cotten, S. R., Ono, H., Quan-Haase, A., Mesch, G., Chen, W., Schulz, J., Hale, T. M., & Stern, M. J. (2015). Digital inequalities and why they matter. *Information, Communication & Society*, 18(5), 569–582.

Roodman, D. (2009). How to do xtabond2: an introduction to difference and system GMM in Stata. *Stata Journal*, 9(1), 86–136.

Sen, A. (1985). *Commodities and Capabilities*. North-Holland.

Sen, A. (1999). *Development as Freedom*. Oxford University Press.

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour*, 4(11), 1208–1214.

Suchman, L. (2007). *Human–Machine Reconfigurations: Plans and Situated Actions*. Cambridge University Press.

Thomas, M. A. (2010). What do the worldwide governance indicators measure? *European Journal of Development Research*, 22(1), 31–54.

Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.

Zuboff, S. (2019). *The Age of Surveillance Capitalism: The Fight for a Human Future at the New Frontier of Power*. PublicAffairs.
