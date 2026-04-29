# Indicator Validity Audit

This document audits every indicator currently used in the country-year and state-year DCLO builds against the four-layer scheme defined in `02_theoretical_framework.md` (resources / conversion factors / capabilities / functionings). Each indicator is rated for **construct validity**, **measurement validity**, and **panel coverage**, with a recommended action.

## 1. Country-year track

Source: `data/gold/dpi_selected_indicators_by_domain.json` (current selection) and `data/gold/dclo_country_year.csv` (column inventory).

### 1.1 ACC — Connectivity

| Indicator | Source | Captures (target = resources/conversion factors) | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WB_BX.KLT.DINV.WD.GD.ZS (FDI net inflows, % GDP) | World Bank | **None.** This is a capital-account indicator. | **FAIL — category error.** | n/a | 8.7 % | **Drop.** |
| WB_IT.NET.SECR.P6 (Secure servers per million) | World Bank / Netcraft | Resource (secure-web infrastructure). | Weak — indicator is supply-side and skewed by hosting-industry concentration. | OK. | 8.7 % | Keep with a hosting-concentration caveat. |
| WB_IT.NET.USER.ZS (Internet users, % pop) | World Bank / ITU | Resource (use coverage). | Strong. | OK. | 7.8 % | Keep. |
| WB_IT.CEL.SETS.P2 (Mobile cellular subscriptions per 100) | World Bank / ITU | Resource (device penetration). | Weak — multi-SIM inflates rate; fixed broadband ignored. | OK with caveat. | 7.8 % | Replace with ITU `IT.NET.BBND.P2` (fixed broadband) **or** keep both with explicit double-count caveat. |

**Recommended replacement set:**
- ITU mobile-broadband subscriptions per 100 (resource, well-coverage).
- ITU 4G+ population coverage (resource, infrastructure quality).
- ITU price of 1 GB mobile-data as % GNI per capita (conversion factor — affordability).
- GSMA Mobile Internet Connectivity Index (resource + conversion factor).

### 1.2 SKL — Digital literacy and competence

| Indicator | Source | Captures | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WB_SE.TER.ENRR (Tertiary gross enrolment) | World Bank / UNESCO | Resource (general education). | **FAIL — does not target digital skills.** | OK. | 7.8 % | **Drop.** |
| WB_SE.SEC.ENRR (Secondary gross enrolment) | World Bank / UNESCO | Resource (general education). | **FAIL — does not target digital skills.** | OK. | 8.7 % | **Drop.** |

**Recommended replacement set:**
- ITU/UNESCO **SDG 4.4.1** (proportion of adults with basic ICT skills, by skill category) — the canonical indicator. This is non-negotiable for an SKL domain.
- OECD PIAAC ICT-related problem-solving (Tier 2/3) — for OECD comparators.
- ILOSTAT digital-skills employment proxy (share of ICT occupations).
- *India track:* NSS-PLFS module on internet use and skills; ASER digital-skills supplement when published.

### 1.3 SRV — Access to essential services through digital channels

| Indicator | Source | Captures | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WTO_SERVICES_TOTAL_EXPORTS | WTO | Macro-economic services-trade. | **FAIL — category error: a country's services exports are unrelated to whether households can access services digitally.** | OK as a trade indicator. | high | **Drop.** |
| WTO_SERVICES_TRANSPORT_EXPORTS | WTO | Macro-economic. | **FAIL.** | OK. | high | **Drop.** |
| WTO_SERVICES_TRAVEL_EXPORTS | WTO | Macro-economic. | **FAIL.** | OK. | high | **Drop.** |
| WTO_SERVICES_POSTAL_COURIER_EXPORTS | WTO | Macro-economic. | **FAIL.** | OK. | high | **Drop.** |

**Recommended replacement set:**
- World Bank **GovTech Maturity Index (GTMI)** — country-year, four pillars (CGSI, PSDI, CEI, GTEI). The first comprehensive cross-country e-government measurement.
- UN **E-Government Development Index (EGDI)** — biennial.
- World Bank **Findex** indicators: digital-payment in past year, account use in past year, mobile money account.
- WHO **Global Strategy on Digital Health monitoring framework** indicators (selected).
- *India track:* CoWIN/eSanjeevani service-utilisation rates; DigiLocker monthly active users; UMANG service-uptake; UPI per-capita transaction count; Aadhaar-authentications per beneficiary scheme.

### 1.4 AGR — Agency, safety, rights

| Indicator | Source | Captures | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WB_CC.PER.RNK (Control of Corruption percentile) | World Bank WGI | Macro-political. | **FAIL — measures macro-institutional quality, not digital agency.** | Contested in PolSci lit (Apaza 2009; Thomas 2010; Langbein & Knack 2010). | 8.7 % | **Drop.** |
| WB_PV.PER.RNK (Political Stability percentile) | World Bank WGI | Macro-political. | **FAIL.** | Contested. | 8.7 % | **Drop.** |

**Recommended replacement set:**
- **Freedom House — Freedom on the Net** sub-scores (Obstacles to Access; Limits on Content; Violations of User Rights).
- **GSMA Mobile Gender Gap** (capability — gendered freedom).
- **Findex female-controlled digital account** (capability — gendered).
- **Article 19 / RSF Index** for press / digital expression.
- **APC Internet Rights Charter** signatory or equivalent national-law indicator.
- **SDG 16.10.2** (countries that adopt and implement constitutional, statutory, and/or policy guarantees for public access to information).
- **Privacy International — Data Protection laws status** index.

### 1.5 ECO — Economic participation

| Indicator | Source | Captures | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WB_SL.TLF.CACT.ZS (LFPR) | World Bank / ILO | Macro-employment, not digital. | **FAIL — not digital-economy specific.** | OK. | 9.5 % | **Drop.** |
| WB_SL.UEM.TOTL.ZS (Unemployment) | World Bank / ILO | Macro-employment. | **FAIL.** | OK. | 9.5 % | **Drop.** |
| WB_NE.TRD.GNFS.ZS (Trade % GDP) | World Bank | Macro-economic. | **FAIL.** | OK. | 8.7 % | **Drop.** |
| WB_NY.GDP.PCAP.KD.ZG (GDP per capita growth) | World Bank | Macro-economic. | **FAIL.** | OK. | 8.7 % | Use as a control, not as ECO. |

**Recommended replacement set:**
- **Findex** — made or received digital payment in past year.
- **ILOSTAT** — share of employment in digitally-mediated occupations (ISCO ICT specialists; digital service occupations).
- **OECD ICT Sector** — share of ICT value added in GVA (where available).
- **UNCTAD Digital Economy Report** indicators (e-commerce participation).
- **Online Labour Index** (Oxford Internet Institute) — gig labour participation.
- *India track:* MSME registrations through Udyam portal; eNAM agricultural digital-transactions; share of GST registrations digital-only; UPI transaction per-capita.

### 1.6 OUT — Realised functionings

| Indicator | Source | Captures | Construct validity | Measurement validity | Panel coverage | Recommendation |
|---|---|---|---|---|---|---|
| WB_SP.POP.GROW (Population growth) | World Bank | Demographics. | **FAIL — no plausible mechanism from digital capability.** | OK. | 8.7 % | **Drop.** Use as a placebo outcome (`05_robustness_protocol.md`). |
| WB_AG.LND.AGRI.ZS (Agricultural land share) | World Bank | Land use. | **FAIL.** | OK. | 8.7 % | **Drop.** Use as a placebo outcome. |
| WB_EG.CFT.ACCS.ZS (Access to clean cooking fuels) | World Bank | Energy access functioning — **only weakly tied to digital capability**. | Weak. | OK. | 8.7 % | Demote to context covariate. |

**Recommended replacement set (functionings):**
- **Findex** — used a financial-account through a digital channel in past year.
- **WHO** — proportion receiving an essential health service through a digital channel.
- **UNESCO** — proportion of secondary-school students who completed a course through a digital channel.
- **UN E-Participation Index** — proportion submitting a public-service request digitally.
- *India track:* Aadhaar-authentication-supported service uptake (per scheme); DBT receipt by households; eShram registrations.

## 2. India state-year track

Source: `data/gold/dclo_state_year.csv`.

| Indicator | Capture | Construct validity | Notes | Recommendation |
|---|---|---|---|---|
| ACC_pop_hh_elec | Resource (electricity) | Weak — electricity ≠ digital connectivity. | Use as conversion factor. | Demote; add NFHS internet-use indicator. |
| SKL_fem_literacy | Conversion factor (general literacy). | Weak — not digital. | Keep as conversion factor; **add** SDG 4.4.1-style ICT skills item from NSS-PLFS. |
| SRV_pop_hh_sf | Functioning (sanitation). | **FAIL** — sanitation is not digital service access. | **Drop or move to context.** Replace with PMJAY digital-claims share, eSanjeevani teleconsults per capita. |
| AGR_fem_15_24_hyg_period | Functioning (menstrual hygiene). | **FAIL** — not digital agency. | **Drop or move to context.** Replace with female mobile ownership (NFHS), female bank-account control. |
| AGR_shg_member_scale | Conversion factor (financial inclusion). | Weak. | Keep with caveat. |
| ECO_prop_hh_microfin | Conversion factor (microfinance access). | Weak. | Keep, supplement with UPI per-capita usage. |
| OUT_hh_income_monthly | Functioning (income). | OK as a general functioning, weakly digital. | Keep but recognise as not-digital-specific. |
| OUT_prop_saving | Functioning (saving habit). | Weak. | Keep with caveat. |
| NAT_upi_total_vol, NAT_upi_total_val, NAT_upi_p2p_vol, NAT_upi_p2m_vol | National aggregate | National, applied uniformly to all states; **does not vary across states** in the same year. | **Move to year-fixed context** — currently included in state-year table as a national constant, contributing nothing to within-year state variation. |
| NAT_ib_no_of_transactions, NAT_ib_amt_of_transactions, NAT_ib_active_users | National aggregate | Same. | Same. |

### 2.1 Indicators that should be ingested for the state track

| Indicator | Source | Layer | Update cadence |
|---|---|---|---|
| Internet users (rural / urban) by state | TRAI Performance Indicator Reports | Resource | Quarterly |
| State-wise UPI transactions per capita | NPCI state dashboards | Functioning (digital-economic) | Monthly |
| State-wise BHIM / BharatNet uptake | DoT / BharatNet | Resource | Quarterly |
| State-wise CSC (Common Service Centre) operations | CSC e-Governance Services | Resource | Monthly |
| State-wise Aadhaar authentication transactions | UIDAI dashboards | Functioning | Daily |
| State-wise eSanjeevani teleconsults | MoHFW | Functioning | Monthly |
| State-wise DigiLocker registrations | NeGD | Functioning | Monthly |
| State-wise PFMS / DBT transfers | PFMS | Functioning | Monthly |
| State-wise eNAM trade volume | eNAM portal | Functioning | Daily |
| State-wise PMJAY claims share digital | NHA | Functioning | Monthly |
| State-wise GST registrations | CBIC | Functioning (formalisation) | Monthly |
| Mobile ownership and use, female | NFHS 5; ASER | Conversion factor (gendered) | Multi-year |
| Internet use in past 30 days | NSS-PLFS module | Capability | Quarterly |
| ICT skills (SDG 4.4.1) | MoSPI; ASER | Capability | Annual |
| Cyber-fraud reports per million | NCRB; I4C | Negative AGR | Annual |
| RTI online filings per capita | DoPT | Capability — voice | Annual |

These exist and are public. The state track has not yet been refreshed against them.

## 3. Summary verdict

Of the **20 unique indicators selected** by the country-track build, **16 fail construct validity** under the capability-grounded scheme. Of the **17 unique indicators** in the state-track gold table, at least **6 fail construct validity** and **7 are national-level constants applied identically across states**.

The current rankings, headline TWFE coefficient, and method-comparison tables are therefore not interpretable as a measure of digital capability. They are interpretable as a measure of **macro-economic structure with a digital-access overlay** — which is a different, less interesting quantity.

The fix is not minor: it requires re-ingesting from a different source set (ITU IDI Annual, Findex, GovTech Maturity Index, GSMA, Freedom House FOTN, ILOSTAT digital-occupations, EGDI, NSS-PLFS, NFHS, NPCI, UIDAI dashboards, NCRB) and rebuilding the gold tables. That is the work of the next sprint, and it is documented in `12_known_issues.md`.

## 4. Acceptance criteria for a paper-ready indicator set

Each indicator in the next release must satisfy:

1. **Layer-targeted:** clearly hits one of {resources, conversion factors, capabilities, functionings}, named in a per-indicator metadata field.
2. **Construct-defensible:** has a documented mechanism by which it indicates the layer of the targeted domain (one paragraph in the indicator-mapping doc).
3. **Cross-source-validated:** wherever possible, has at least one alternative source for triangulation (e.g., ITU and Findex for digital-payment use).
4. **Coverage:** ≥ 0.40 panel-cell coverage (entity × year), not just per-indicator pct_observed.
5. **Decomposable:** disaggregatable by gender or by urban-rural where the underlying source permits.
6. **Stable:** no methodology change in the source within the panel window without a discontinuity flag.
7. **Open:** licence permits redistribution under the project's CC-BY licence, or aggregation rules are documented.

Indicators that fail any of (1)–(4) are excluded; indicators that fail (5)–(7) are flagged.
