# DRW LLM Unlearning Project

## Directory Setup

- `filter_tcn.py` requires the Transformer ConceptNet parts to be located in a dir called `tcn`
- To filter based on a series of cluster ids they should be stored in a json list like `furniture-seating-clusters-ids.json` 

├── filter_tcn.py
├── furniture-seating-clusters-ids.json
├── README.md
├── tcn
│   ├── albert-base-v1
│   ├── bert-base-cased
│   ├── roberta-base
│   ├── xlm-roberta-base
│   └── xlnet-base-cased
├── tcn.zip
├── utils.py
└── venv