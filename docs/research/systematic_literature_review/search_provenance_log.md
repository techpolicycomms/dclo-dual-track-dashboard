# Search Provenance Log

## Scopus

**Use:** formal academic database search through the user's JGU academic access.  
**Focused query:** `(("digital capability" OR "digital capabilities") AND ("life outcomes" OR employment OR employability OR health OR wellbeing OR "well-being" OR education OR learning OR "quality of life" OR "social inclusion" OR "social participation"))`  
**Output used:** raw 566-record CSV export preserved in `raw_exports/`; 10 relevance-ranked records logged as Scopus seed records for the evidence map.  
**Guardrail:** final inclusion requires DOI/publisher/library verification, de-duplication, title/abstract screening, full-text review, and quality appraisal.

## Supplementary discovery platform A

**Use:** seed discovery for digital capability and life-outcome literature.  
**Query:** `digital capabilities digital literacy life outcomes employment health education social inclusion`  
**Output used:** 20 visible candidate references logged as supplementary seed records.  
**Guardrail:** these records have no synthesis weight until independently verified.

## Supplementary discovery platform B

**Use:** supplementary seed discovery in an authenticated browser session.  
**Output used:** 10 visible candidate records logged as supplementary seed records.  
**Important record surfaced:** Fisk et al. (2022), "Healing the Digital Divide With Digital Inclusion: Enabling Human Capabilities," Journal of Service Research, DOI: 10.1177/10946705221140148.  
**Guardrail:** repository, preprint, or generic-looking records require stricter quality appraisal before inclusion.

## Supplementary discovery platform C

**Use:** attempted as a supplementary discovery route.  
**Outcome:** no usable result export was visible in the session.  
**Next step:** rely on Scopus and other formal database exports for final PRISMA counts.

## Citation Rule

Discovery platforms can identify leads, but they cannot establish citation truth. Final citations must be verified against source documents, DOI/Crossref, publisher pages, official PDFs, or library records.
