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
  data.py        caricamento, pulizia, nomi colonne, identifier sintetici
  risk.py        sample uniqueness, k per classe di equivalenza, record linkage
  utility.py     baseline di utilità (logistic regression: accuracy, AUC)
  protection.py  tecniche di masking (non perturbative / perturbative)
notebooks/       analisi passo-passo (EDA, rischio, protezione, R-U map)
figures/         grafici esportati per report e slide
report/          report finale (template PPIA_HOMEWORK_1)
slides/          presentazione 7+3 min (template PPIA_HW1)
PROMPT_LOG.md    log dei prompt GenAI (richiesto per la disclosure)
```


## Scadenze
- 25/10/2026 consegna – 26/10/2026 presentazione
