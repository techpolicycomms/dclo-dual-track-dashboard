"""Seed the knowledge base from existing project documentation and data.

Ingests docs/*.md and key data files into kb/raw/, then compiles
the initial wiki. This bootstraps the KB from the DCLO project's
existing content.

Usage:
    python -m src.kb seed
"""

from pathlib import Path

from .config import load_config, get_root
from .ingest import ingest_file, ingest_directory
from .compile import compile_wiki
from .index import rebuild_index, rebuild_catalog


# Mapping of existing docs to categories
_DOC_CATEGORIES = {
    "dclo-results-brief.md": "findings",
    "dclo-data-catalog.md": "data",
    "dclo-causal-methodology.md": "methodology",
    "dclo-identification-strategy.md": "methodology",
    "dclo-model-governance.md": "governance",
    "dclo-indicator-mapping.md": "data",
    "dclo-idp-granular-data-plan.md": "data",
    "dpi-indicator-intake.md": "data",
    "dclo-dataset-intake-template.md": "data",
    "data-contract.md": "governance",
    "methodology-formative-sem.md": "methodology",
    "dashboard-explainer.md": "dashboard",
    "dashboard-spec.md": "dashboard",
    "hosting-guide.md": "operations",
    "runbook.md": "operations",
}

_DOC_TAGS = {
    "dclo-results-brief.md": ["dclo", "results", "findings"],
    "dclo-data-catalog.md": ["dclo", "data", "catalog"],
    "dclo-causal-methodology.md": ["dclo", "causal", "econometrics"],
    "dclo-identification-strategy.md": ["dclo", "causal", "identification"],
    "dclo-model-governance.md": ["dclo", "governance", "indicators"],
    "dclo-indicator-mapping.md": ["dclo", "indicators", "mapping"],
    "data-contract.md": ["data", "contract", "governance"],
    "methodology-formative-sem.md": ["methodology", "sem", "formative"],
    "dashboard-explainer.md": ["dashboard", "user-guide"],
    "dashboard-spec.md": ["dashboard", "power-bi", "spec"],
    "runbook.md": ["operations", "pipeline"],
}


def main():
    cfg = load_config()
    root = get_root()
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Seeding Knowledge Base")
    print("=" * 50)
    print()

    # 1. Ingest existing documentation
    docs_dir = root / "docs"
    if docs_dir.exists():
        print("Phase 1: Ingesting docs/*.md ...")
        for md_path in sorted(docs_dir.glob("*.md")):
            category = _DOC_CATEGORIES.get(md_path.name, "documentation")
            tags = _DOC_TAGS.get(md_path.name, ["dclo"])
            ingest_file(
                md_path, raw_dir,
                source=f"project:docs/{md_path.name}",
                tags=tags,
                category=category,
            )
        print()

    # 2. Ingest gold data catalog (CSV metadata, not full data)
    gold_dir = root / "data" / "gold"
    if gold_dir.exists():
        print("Phase 2: Ingesting data/gold/ summaries ...")
        for csv_path in sorted(gold_dir.glob("*.csv")):
            ingest_file(
                csv_path, raw_dir,
                source=f"project:data/gold/{csv_path.name}",
                tags=["dclo", "data", "gold"],
                category="data",
            )
        for json_path in sorted(gold_dir.glob("*.json")):
            ingest_file(
                json_path, raw_dir,
                source=f"project:data/gold/{json_path.name}",
                tags=["dclo", "data", "gold"],
                category="data",
            )
        print()

    # 3. Ingest config files
    config_dir = root / "config"
    if config_dir.exists():
        print("Phase 3: Ingesting config files ...")
        for yml_path in sorted(config_dir.glob("*.yml")):
            if yml_path.name == "kb.yml":
                continue  # Skip our own config
            if "example" in yml_path.name:
                continue
            ingest_file(
                yml_path, raw_dir,
                source=f"project:config/{yml_path.name}",
                tags=["dclo", "config"],
                category="configuration",
            )
        print()

    # 4. Ingest README
    readme = root / "README.md"
    if readme.exists():
        print("Phase 4: Ingesting README ...")
        ingest_file(
            readme, raw_dir,
            source="project:README.md",
            tags=["dclo", "overview"],
            category="overview",
        )
        print()

    # 5. Compile the wiki
    print("Phase 5: Compiling wiki ...")
    compile_wiki(cfg, full=True)
    print()

    # 6. Build index and catalog
    print("Phase 6: Building index and catalog ...")
    rebuild_index(cfg)
    rebuild_catalog(cfg)
    print()

    print("=" * 50)
    print("  Seed complete! Your knowledge base is ready.")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  python -m src.kb search 'your query'")
    print("  python -m src.kb lint")
    print("  python -m src.kb output report --topic 'DCLO methodology'")
    print("  python -m src.kb index --stats")


if __name__ == "__main__":
    main()
