# CLAUDE.md — PPIA Homework 1 (Microdata protection)

## 1. A cosa serve questo repository
Contiene **solo codice, esperimenti e risultati** dell'Homework 1 del corso
*Privacy Preserving Information Access* (UniPD, A.Y. 2026/2027).
Report e slide si scrivono **fuori dal repository** (editor esterni): qui si
producono i numeri, le tabelle, le figure e gli snippet di codice che servono
a scriverli. Non creare file di report o di presentazione.

- Consegna: **25/10/2026** — presentazione: **26/10/2026** (7 min + 3 di domande).
- Lo studente comunica in italiano. Codice, commenti, figure e tabelle in inglese.
- **Perimetro teorico: solo la lezione "Macrodata and Microdata protection".**
  Non usare k-anonymity, ℓ-diversity, t-closeness, Mondrian, differential privacy
  (lezioni successive): al massimo vanno citate come sviluppi futuri nel report.

## 2. Obiettivi (cosa il lavoro deve dimostrare)
1. **Rimuovere gli identifier non basta.** Dopo il sanitizing, misurare quanto
   sono re-identificabili i record usando solo i quasi-identifier (uniqueness +
   record linkage contro un dataset esterno simulato) e mostrare che i più
   esposti sono gli outlier (highly visible tuples).
2. **Masking non perturbativo**: global recoding, top/bottom coding,
   generalization, local suppression → quanto scende il rischio e quanta
   informazione si perde.
3. **Masking perturbativo**: PRAM, rumore additivo, rank swapping,
   micro-aggregation → stesso confronto.
4. **Attribute disclosure**: anche senza identificare il record, alcuni gruppi
   di QI hanno `default` tutto uguale → misurare quanti record sono esposti e
   come cambia dopo la protezione.
5. **Trade-off rischio/utilità**: ogni configurazione valutata su rischio,
   information loss e utilità (AUC di un classificatore di `default`) e
   riassunta nella **R-U confidentiality map**.

Priorità: 1 → 2 → 3 → 5 sono essenziali; 4 e gli opzionali (sezione 8) solo se
avanza tempo. Meglio pochi esperimenti ben spiegati che molte tecniche elencate.

## 3. Output finali attesi
Tutto riproducibile da zero con `python main.py` (nessun notebook).

**Tabelle** in `results/` (CSV):
- `attribute_classification.csv` — colonna, ruolo (identifier / QI / sensitive / other), motivazione
- `baseline.csv` — uniqueness, linkage rate (deterministico e distance-based),
  attribute disclosure rate, AUC e accuracy sui dati originali
- `experiments.csv` — una riga per configurazione (inclusa la riga `original`,
  così la R-U map legge un solo file): famiglia (non perturbative / perturbative),
  tecnica, parametri, uniqueness, linkage rate, attribute disclosure rate,
  information loss (continuo e categorico), AUC, ΔAUC, % record soppressi, tempo.
  **È la tabella principale**: R-U map e conclusioni derivano da qui.

**Figure** in `figures/` (PNG 300 dpi, leggibili su una slide):
- `qi_distributions.png` — distribuzioni dei QI e del target
- `qi_combination_sizes_original.png` — quanti record hanno una combinazione di QI
  unica / rara (QI base e QI estesi)
- `risk_by_technique.png` — uniqueness e linkage rate per tecnica e parametro
- `utility_by_technique.png` — AUC e information loss per tecnica e parametro
- `ru_map.png` — rischio (asse y) vs utilità (asse x), un punto per configurazione,
  più i punti `original` e `no data` e la linea di massimo rischio tollerabile

**Snippet per le slide**: le funzioni chiave (`sample_uniqueness`,
`record_linkage`, `local_suppression`, `pram`, `microaggregation`) devono essere
brevi e leggibili, così da poterle mostrare in 2–3 slide.

**Numeri chiave** stampati a fine `03_experiments.py` e salvati in
`results/summary.md`: % record unici prima/dopo, linkage rate prima/dopo,
configurazione scelta come compromesso, perdita di AUC corrispondente.

## 4. Dataset
UCI *Default of Credit Card Clients* (id 350): 30.000 clienti di una banca
taiwanese, apr–set 2005. File in `data/raw/` (non versionato).
Il `.xls` ha i nomi reali nella seconda riga → `header=1`.

| Ruolo | Colonne |
|---|---|
| Identifier | `ID` + sintetici `customer_code`, `phone` (simulano lo scenario reale, poi rimossi con `sanitize`) |
| Quasi-identifier | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Sensibili | `default` (principale), `LIMIT_BAL`, `PAY_0`, `PAY_2..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6` |

- Scenario d'avversario esteso: chi conosce anche `LIMIT_BAL` approssimato
  (es. dipendente della banca) → QI estesi `QI + [LIMIT_BAL]`.
- Codici non documentati (`EDUCATION` 0/5/6, `MARRIAGE` 0) accorpati in "others" da `clean()`.
- Target sbilanciato (~22% default): usare **AUC**, non solo accuracy.
- Il dataset è un campione (clienti di una banca), non la popolazione: la
  uniqueness calcolata è **sample uniqueness** e va detto nel report.

## 5. Definizioni di riferimento (dalla lezione)
- **Identifier**: identifica (quasi) univocamente. **Quasi-identifier**: da solo
  dice poco, ma combinato con altri può identificare. **Confidential**: riguarda
  la sfera personale. **Non-confidential**: il resto.
- **Sanitizing**: rimozione degli identifier espliciti.
- **Disclosure**: identity (record re-identificato), attribute (valore sensibile
  dedotto anche senza identificare), inferential (dedotto con alta probabilità),
  membership (si scopre che la persona è nel dataset).
- **Uniqueness**: Pr(PU) = Σ_j I(F_j = 1) / N, con j combinazioni dei QI,
  F_j numero di record con quella combinazione, N numero di record.
- **Record linkage**: l'avversario unisce il rilascio con un dataset esterno
  sui QI. Deterministico (join esatto; con m match il record è indovinato con
  prob. 1/m) o distance-based (nearest neighbour con distanza pesata sugli
  attributi standardizzati). Linkage rate = % di record re-identificati correttamente.
- **Attribute disclosure rate** (misura nostra, dalla definizione di slide 42):
  % di record che stanno in un gruppo di QI con ≥ 2 record e `default` costante.
- **Masking non perturbativo**: sampling, global recoding, top/bottom coding,
  generalization, local suppression (sostituisce con missing i QI delle
  combinazioni rare).
- **Masking perturbativo**: rounding, resampling, PRAM (matrice di transizione
  tra categorie), rumore additivo (scorrelato / correlato), swapping, rank
  swapping (scambi entro distanza p di rango), micro-aggregation (cluster di
  record simili, valore sostituito dalla media/moda del cluster), blank and impute.
- **Sintesi**: fully synthetic (es. Cholesky: X_syn = R·U con C = Uᵀ·U) o
  parzialmente sintetico.
- **Information loss**: continuo → MSE, MAE, variazione di media/varianza;
  categorico → % record cambiati (distanza in categorie se ordinati), confronto
  delle tabelle di contingenza, entropia; ML → calo di AUC/accuracy.
- **R-U map**: utilità (1/information loss) sull'asse x, rischio sull'asse y,
  soglia di massimo rischio tollerabile.

## 6. Gerarchie di generalizzazione (livello 0 = originale)
Usate da global recoding / generalization, un livello scelto per attributo
(nessun algoritmo di ricerca del livello ottimo).
- `AGE`: esatta → fasce 5 anni → fasce 10 anni → {<30, 30–50, >50}
- `EDUCATION`: graduate/university/high school/others → {higher, other}
- `MARRIAGE`: married/single/others → {married, not married}
- `LIMIT_BAL` (solo scenario esteso): fasce 50k → 100k → 200k

## 7. Struttura
```
src/data.py         caricamento, pulizia, identifier sintetici, sanitize  (FATTO)
src/hierarchies.py  gerarchie e generalize(df, levels)
src/risk.py         sample uniqueness, record linkage (deterministico e
                    distance-based), attribute disclosure rate
src/utility.py      classificatore (AUC), information loss continuo e categorico
src/protection.py   masking non perturbativo e perturbativo
scripts/01_eda.py          statistiche descrittive (stampate) + qi_distributions.png,
                           qi_combination_sizes_original.png, attribute_classification.csv
scripts/02_baseline.py     rischio e utilità sui dati originali → baseline.csv
scripts/03_experiments.py  tutte le configurazioni → experiments.csv, summary.md
scripts/04_figures.py      risk_by_technique, utility_by_technique, ru_map
                           (legge solo i CSV: si rifanno le figure senza rieseguire gli esperimenti)
main.py             esegue gli script in ordine; opzione per lanciarne uno solo
results/            tabelle CSV + summary.md (sezione 3)
figures/            figure PNG (sezione 3)
```

## 8. Esperimenti
- Dataset esterno dell'avversario: campione casuale (es. 50%) dei clienti con
  QI originali e `ID`; il linkage si fa contro il rilascio protetto.
- Ogni configurazione con QI base e QI estesi.
- Non perturbative:
  - global recoding di `AGE` (fasce 5 / 10 / 3 fasce)
  - top/bottom coding di `AGE` (es. <25, >60) e `LIMIT_BAL` (es. >500k)
  - generalization di `EDUCATION`, `MARRIAGE` (livello 1)
  - local suppression delle combinazioni con frequenza < soglia (soglia ∈ {2, 5, 10})
- Perturbative (2–3 livelli di parametro ciascuna):
  - PRAM su `EDUCATION`, `MARRIAGE` (prob. di mantenere il valore ∈ {0.9, 0.7, 0.5})
  - rumore additivo su `AGE`, `LIMIT_BAL` (σ ∈ {0.05, 0.1, 0.2} · std)
  - rank swapping su `AGE`, `LIMIT_BAL` (p ∈ {5, 10, 20}%)
  - micro-aggregation su `AGE` + `LIMIT_BAL` (cluster di dimensione ∈ {3, 5, 10})
- 1–2 combinazioni (es. recoding `AGE` + generalization + local suppression).
- Classificatore: logistic regression, stesso split (`random_state=42`, 70/30
  stratificato) per tutte le versioni; categorie codificate one-hot, missing
  come categoria a sé. Nota attesa: proteggere solo i QI incide poco sull'AUC
  (sono poco predittivi), perturbare `LIMIT_BAL` incide di più.
- Opzionali: sampling (effetto sulla membership); resampling; blank and impute;
  dati sintetici con Cholesky; **macrodata** (`scripts/05_macrodata.py`):
  tabella dei conteggi di `default` per fascia d'età × educazione con threshold
  rule, cell suppression, rounding, roll-up, CTA, e tabella di magnitudo (media
  `LIMIT_BAL`) con (n,k)-rule e p-percentage.

## 9. Convenzioni e comandi
- Python 3.11+, pandas/numpy/scipy/scikit-learn/seaborn; virtualenv `.venv/`.
- Seed `42` ovunque. Logica in funzioni pure in `src/`; gli script in `scripts/`
  orchestrano, stampano un riepilogo leggibile e salvano CSV/figure.
- Ogni script eseguibile da solo dalla root: `python scripts/0X_nome.py`.
- Non committare mai `data/raw/`.
```bash
bash setup.sh                                               # primo setup
source .venv/bin/activate
python -c "from src.data import load; print(load().shape)"  # verifica dati
python main.py                                              # tutta la pipeline
python scripts/04_figures.py                                # solo le figure
```

## 10. Regole di Routing del Modello
- Per modifiche semplici, script rapidi o analisi di log estesi: CAMBIA MODELLO e usa HAIKU (risparmio token 80%).
- Per refactoring complessi o bug introvabili: Richiedi l'uso di SONNET o OPUS.
- Se il limite è vicino al 90%, avvisami per compattare la chat (`/clear`) prima del blocco.