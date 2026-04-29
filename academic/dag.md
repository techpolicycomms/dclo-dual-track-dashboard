# DAG — Identification Strategy

The directed acyclic graph below operationalises the estimand stated in `04_identification_strategy_revised.md` §1. It is reproduced here so the dashboard can render it inline.

## Mermaid (rendered by the Streamlit dashboard)

```mermaid
flowchart LR
    classDef closed fill:#26A69A,color:#fff,stroke:#00897B
    classDef open fill:#EF5350,color:#fff,stroke:#C62828
    classDef control fill:#42A5F5,color:#fff,stroke:#1976D2
    classDef target fill:#7E57C2,color:#fff,stroke:#5E35B2

    nu["ν_i — entity FE<br/>(time-invariant heterogeneity)"]:::closed
    lambda["λ_t — year FE<br/>(global shocks)"]:::closed
    X["X_{i,t} — controls<br/>(GDP per capita,<br/>urbanisation,<br/>demographics, electricity)"]:::control
    R["R_{i,t} — country-year reform<br/>(UPI 2016, India Stack 2017,<br/>Jio price collapse 2016, COVID 2020)"]:::open

    D["D_{i, t-1} — capability composite<br/>(predictor)"]:::target
    Y["Y_{i, t} — functioning achievement<br/>(outcome)"]:::target

    GDP["GDP per capita,<br/>urbanisation,<br/>state capacity"]:::control

    nu --> D
    nu --> Y
    lambda --> D
    lambda --> Y
    GDP --> D
    GDP --> Y
    R -.-> D
    R -.-> Y
    X --> D
    X --> Y
    D -->|"β = estimand"| Y
```

## Reading the DAG

- **Solid arrows from controls** (entity FE, year FE, X) are paths the estimator closes.
- **Dotted arrows from R** are paths the estimator does **not** close. R is a country-year-specific reform that drives both D in the prior period and Y in the current period; two-way fixed effects do not absorb it.
- The **β arrow from D to Y** is the estimand defined in `04_identification_strategy_revised.md` §1.

## Why this matters for the current build

Construct contamination (predictor and outcome share constituent indicators) operates *additionally* to the DAG threats above. The DAG is correct only after the disjoint-source partition of `04_identification_strategy_revised.md` §3.4 is implemented. Until then, the published β is biased by both (a) the unclosed R-path and (b) construct overlap.

## How the dashboard uses this

The Causal Evidence tab embeds the mermaid block inline, with a one-paragraph explainer keyed to the threats listed in `04_identification_strategy_revised.md` §3 and a callout reading: "β = 0.624 should be read in light of these threats; see academic/04_identification_strategy_revised.md §5."
