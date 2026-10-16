# PPIA – Homework 1: Microdata protection on *Default of Credit Card Clients*

Privacy Preserving Information Access – Università di Padova, A.Y. 2026/2027

Analisi del rischio di re-identificazione e applicazione di tecniche di protezione
dei microdati al dataset UCI *Default of Credit Card Clients* (Yeh & Lien, 2009),
con misura del trade-off rischio/utilità tramite R-U confidentiality map.

## Dataset
- Fonte: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- 30.000 clienti di una banca taiwanese (apr–set 2005), 23 feature + target `default`.
- Licenza: CC BY 4.0. Il file grezzo **non** è versionato: vedi `data/README.md`.

## Struttura
```
src/
  data.py         caricamento, pulizia, identifier sintetici, sanitize
  hierarchies.py  gerarchie per global recoding / generalization
  risk.py         sample uniqueness, record linkage, attribute disclosure rate
  utility.py      classificatore (AUC), information loss continuo e categorico
  protection.py   masking non perturbativo e perturbativo
scripts/          pipeline numerata (01_eda, 02_baseline, 03_experiments, 04_figures)
main.py           esegue la pipeline
results/          tabelle CSV e summary.md
figures/          grafici PNG (300 dpi) per report e slide
```

## Uso
```bash
bash setup.sh                  # primo setup (crea .venv e installa le dipendenze)
source .venv/bin/activate
python main.py                 # tutta la pipeline
python main.py 02              # un solo script
python scripts/04_figures.py   # solo le figure (legge i CSV in results/)
```

## Scadenze
- 25/10/2026 consegna – 26/10/2026 presentazione
