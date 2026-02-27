# DPI Indicator Intake (Auto-generated)

This file classifies indicators for country-year DCLO into `core_formative`, `context_only`, or `exclude` using balanced gates.

## Gating Columns

- `pct_observed_global`
- `imputation_share`
- `years_covered`
- `ranking_score`

## Intake Table (Top 120 rows)

| indicator_code | indicator_name | pillar | years_covered | economies_covered | imputation_share | dclo_domain | pct_observed_global | panel_coverage | ranking_score | role | gate_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WTO_SERVICES_BUSINESS_EXPORTS | Services exports (business) | access_usage | 6 | 47 | 0.0 | SRV | 100.0 | 0.4809688581314879 | 154.29688581314878 | context_only | True |
| WB_BX.KLT.DINV.WD.GD.ZS | Foreign direct investment net inflows (% of GDP) | access_usage | 10 | 5 | 0.0 | ACC | 100.0 | 0.08650519031141868 | 111.65051903114187 | core_formative | False |
| WB_IT.NET.SECR.P6 | Secure Internet servers (per 1 million people) | trust_governance | 10 | 5 | 0.0 | ACC | 100.0 | 0.08650519031141868 | 111.65051903114187 | core_formative | False |
| WB_IT.NET.USER.ZS | Individuals using the Internet (% of population) | access_usage | 11 | 5 | 0.0 | ACC | 100.0 | 0.07785467128027682 | 111.03546712802768 | core_formative | False |
| WB_IT.CEL.SETS.P2 | Mobile cellular subscriptions (per 100 people) | access_usage | 9 | 5 | 0.0 | ACC | 100.0 | 0.07785467128027682 | 110.53546712802768 | core_formative | False |
| WB_CC.PER.RNK | Control of Corruption Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | core_formative | False |
| WB_PV.PER.RNK | Political Stability and Absence of Violence Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | core_formative | False |
| WB_SL.TLF.CACT.ZS | Labor force participation rate total (% of total population ages 15+) | affordability_inclusion | 11 | 5 | 0.0 | ECO | 100.0 | 0.09515570934256055 | 112.76557093425606 | core_formative | False |
| WB_SL.UEM.TOTL.ZS | Unemployment (% of labor force) | affordability_inclusion | 11 | 5 | 0.0 | ECO | 100.0 | 0.09515570934256055 | 112.76557093425606 | core_formative | False |
| WB_NE.TRD.GNFS.ZS | Trade (% of GDP) | access_usage | 10 | 5 | 0.0 | ECO | 100.0 | 0.08650519031141868 | 111.65051903114187 | core_formative | False |
| WB_NY.GDP.PCAP.KD.ZG | GDP per capita growth (annual %) | affordability_inclusion | 10 | 5 | 0.0 | ECO | 100.0 | 0.08650519031141868 | 111.65051903114187 | core_formative | False |
| WB_SP.POP.GROW | Population growth (annual %) | sustainability_resilience | 10 | 5 | 0.0 | OUT | 100.0 | 0.08650519031141868 | 111.65051903114187 | core_formative | False |
| WB_AG.LND.AGRI.ZS | Agricultural land (% of land area) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | core_formative | False |
| WB_EG.CFT.ACCS.ZS | Access to clean fuels and technologies for cooking (% of population) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | core_formative | False |
| WB_SE.ADT.LITR.ZS | Literacy rate adult total (% of people ages 15 and above) | affordability_inclusion | 6 | 2 | 0.0 | SKL | 100.0 | 0.012110726643598616 | 102.91107266435986 | core_formative | False |
| WTO_SERVICES_POSTAL_COURIER_EXPORTS | Services exports (postal and courier) | trust_governance | 11 | 47 | 0.0 | SRV | 100.0 | 0.8944636678200693 | 196.8963667820069 | core_formative | True |
| WTO_SERVICES_TOTAL_EXPORTS | Services exports (total) | access_usage | 11 | 47 | 0.0 | SRV | 100.0 | 0.8944636678200693 | 196.8963667820069 | core_formative | True |
| WTO_SERVICES_TRANSPORT_EXPORTS | Services exports (transport) | access_usage | 11 | 47 | 0.0 | SRV | 100.0 | 0.8944636678200693 | 196.8963667820069 | core_formative | True |
| WTO_SERVICES_TRAVEL_EXPORTS | Services exports (travel) | access_usage | 11 | 47 | 0.0 | SRV | 100.0 | 0.8944636678200693 | 196.8963667820069 | core_formative | True |
| WB_IT.NET.BBND.P2 | Fixed broadband subscriptions (per 100 people) | access_usage | 9 | 5 | 0.0 | ACC | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_TM.VAL.MRCH.XD.WD | Import value index (2015 = 100) | access_usage | 9 | 5 | 0.0 | ACC | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_GE.PER.RNK | Government Effectiveness Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_RL.PER.RNK | Rule of Law Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_RQ.PER.RNK | Regulatory Quality Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_VA.PER.RNK | Voice and Accountability Percentile Rank | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_FB.BNK.CAPA.ZS | Bank capital to assets ratio | trust_governance | 9 | 5 | 0.0 | AGR | 100.0 | 0.07439446366782007 | 110.18944636678201 | exclude | False |
| WB_NY.GNP.PCAP.CD | GNI per capita (current US$) | affordability_inclusion | 10 | 5 | 0.0 | ECO | 100.0 | 0.08650519031141868 | 111.65051903114187 | exclude | False |
| WB_SP.URB.TOTL.IN.ZS | Urban population (% of total population) | affordability_inclusion | 10 | 5 | 0.0 | ECO | 100.0 | 0.08650519031141868 | 111.65051903114187 | exclude | False |
| WB_SE.TER.ENRR | School enrollment tertiary (% gross) | affordability_inclusion | 10 | 5 | 0.0 | ECO | 100.0 | 0.07958477508650519 | 110.95847750865052 | exclude | False |
| WB_SP.DYN.LE00.IN | Life expectancy at birth (years) | affordability_inclusion | 9 | 5 | 0.0 | ECO | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_SE.SEC.ENRR | School enrollment secondary (% gross) | affordability_inclusion | 10 | 4 | 0.0 | ECO | 100.0 | 0.06055363321799308 | 108.95536332179931 | exclude | False |
| WB_AG.LND.FRST.ZS | Forest area (% of land area) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_EG.ELC.ACCS.ZS | Access to electricity (% of population) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_EG.USE.ELEC.KH.PC | Electric power consumption (kWh per capita) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_EG.USE.PCAP.KG.OE | Energy use (kg of oil equivalent per capita) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_EN.POP.DNST | Population density (people per sq. km of land area) | sustainability_resilience | 9 | 5 | 0.0 | OUT | 100.0 | 0.07785467128027682 | 110.53546712802768 | exclude | False |
| WB_ER.H2O.FWTL.ZS | Annual freshwater withdrawals (% of internal resources) | sustainability_resilience | 8 | 5 | 0.0 | OUT | 100.0 | 0.06920415224913495 | 109.4204152249135 | exclude | False |
| WB_EG.FEC.RNEW.ZS | Renewable energy consumption (% of total final energy) | sustainability_resilience | 7 | 5 | 0.0 | OUT | 100.0 | 0.06055363321799308 | 108.3053633217993 | exclude | False |
| WB_EN.ATM.PM25.MC.M3 | PM2.5 air pollution mean annual exposure (micrograms per cubic meter) | sustainability_resilience | 6 | 5 | 0.0 | OUT | 100.0 | 0.05190311418685121 | 107.19031141868513 | exclude | False |
| EDGAR_CO2_TOTAL_KT | EDGAR CO2 emissions total (kt CO2) | sustainability_resilience | 11 | 5 | 0.0 | OUT | 0.0 | 0.09515570934256055 | 12.765570934256056 | exclude | False |
| GHG_PROFILE_CH4_KTCO2E | CH4 emissions (kt CO2e, profile) | sustainability_resilience | 8 | 5 | 0.0 | OUT | 0.0 | 0.04498269896193772 | 6.998269896193772 | exclude | False |
| GHG_PROFILE_CO2_KTCO2E | CO2 emissions (kt CO2e, profile) | sustainability_resilience | 8 | 5 | 0.0 | OUT | 0.0 | 0.04498269896193772 | 6.998269896193772 | exclude | False |
| GHG_PROFILE_N2O_KTCO2E | N2O emissions (kt CO2e, profile) | sustainability_resilience | 8 | 5 | 0.0 | OUT | 0.0 | 0.04498269896193772 | 6.998269896193772 | exclude | False |
| GHG_PROFILE_TOTAL_NO_LULUCF_KTCO2E | GHG emissions without LULUCF (kt CO2e, profile) | sustainability_resilience | 8 | 5 | 0.0 | OUT | 0.0 | 0.04498269896193772 | 6.998269896193772 | exclude | False |
| GTMI_CGSI | GovTech Core Government Systems Index | access_usage | 3 | 47 | 0.0 | SRV | 100.0 | 0.24394463667820068 | 129.84446366782007 | exclude | False |
| GTMI_DCEI | GovTech Digital Citizen Engagement Index | trust_governance | 3 | 47 | 0.0 | SRV | 100.0 | 0.24394463667820068 | 129.84446366782007 | exclude | False |
| GTMI_GTEI | GovTech GovTech Enablers Index | trust_governance | 3 | 47 | 0.0 | SRV | 100.0 | 0.24394463667820068 | 129.84446366782007 | exclude | False |
| GTMI_GTMI | GovTech Maturity Index (overall) | access_usage | 3 | 47 | 0.0 | SRV | 100.0 | 0.24394463667820068 | 129.84446366782007 | exclude | False |
| GTMI_PSDI | GovTech Public Service Delivery Index | access_usage | 3 | 47 | 0.0 | SRV | 100.0 | 0.24394463667820068 | 129.84446366782007 | exclude | False |
| WB_BX.GSR.CCIS.ZS | ICT service exports (% of service exports) | access_usage | 10 | 5 | 0.0 | SRV | 100.0 | 0.08650519031141868 | 111.65051903114187 | exclude | False |
| WB_NE.EXP.GNFS.ZS | Exports of goods and services (% of GDP) | access_usage | 10 | 5 | 0.0 | SRV | 100.0 | 0.08650519031141868 | 111.65051903114187 | exclude | False |
| WB_NE.IMP.GNFS.ZS | Imports of goods and services (% of GDP) | access_usage | 10 | 5 | 0.0 | SRV | 100.0 | 0.08650519031141868 | 111.65051903114187 | exclude | False |