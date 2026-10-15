# CLAUDE.md — PPIA Homework 1 (Microdata protection)

## 1. A cosa serve questo repository
Contiene **solo codice, esperimenti e risultati** dell'Homework 1 del corso
*Privacy Preserving Information Access* (UniPD, A.Y. 2026/2027).
Report e slide si scrivono **fuori dal repository** (editor esterni): qui si
producono i numeri, le tabelle, le figure e gli snippet di codice che servono
a scriverli. Non creare file di report o di presentazione.

- Consegna: **25/10/2026** — presentazione: **26/10/2026** (7 min + 3 di domande).
- Lo studente comunica in italiano. Codice, commenti, figure e tabelle in inglese.

## 2. Obiettivi (cosa il lavoro deve dimostrare)
1. **Rimuovere gli identifier non basta.** Misurare quanto sono re-identificabili
   i record usando solo i quasi-identifier (uniqueness + record linkage contro
   un dataset esterno simulato).
2. **k-anonymity protegge dall'identity disclosure**: ottenerla con
   generalization + suppression per vari k e mostrare quanto scende il rischio.
3. **…ma non dall'attribute disclosure.** L'attributo sensibile `default` è
   binario (ℓ ≤ 2) e sbilanciato (~22%): mostrare classi omogenee
   (homogeneity attack) e classi molto lontane dalla distribuzione globale
   (skewness attack) → misurare ℓ-diversity e t-closeness.
4. **Confronto con tecniche perturbative** (PRAM, rumore, micro-aggregation).
5. **Trade-off rischio/utilità**: ogni configurazione valutata su rischio e
   utilità (AUC di un classificatore di `default`) e riassunta nella **R-U map**.

Priorità: 1 → 2 → 3 → 5 sono essenziali; 4 e gli opzionali solo se avanza tempo.
Meglio pochi esperimenti ben spiegati che molte tecniche elencate.

## 3. Output finali attesi
Tutto riproducibile da zero con `python main.py` (nessun notebook).

**Tabelle** in `results/` (CSV):
- `attribute_classification.csv` — colonna, ruolo (identifier / QI / sensitive / other), motivazione
- `baseline.csv` — uniqueness, linkage rate, AUC e accuracy sui dati originali
- `experiments.csv` — una riga per configurazione (incluse le righe `original` di
  baseline, così la R-U map legge un solo file): tecnica, parametri, k effettivo,
  ℓ, t, uniqueness, linkage rate, AUC, information loss, discernibility,
  % soppressi, tempo. **È la tabella principale**: R-U map e conclusioni derivano da qui.

**Figure** in `figures/` (PNG 300 dpi, leggibili su una slide):
- `qi_distributions.png` — distribuzioni dei QI e del target
- `equivalence_class_sizes_original.png` — quanti record sono unici / in classi piccole
- `risk_vs_k.png` — uniqueness e linkage rate al variare di k
- `l_t_vs_k.png` — ℓ e t risultanti al variare di k (con soglia ℓ=2 e base rate)
- `homogeneity_example.png` (o tabella in `results/`) — esempio concreto di classe k-anonima ma omogenea
  Le classi non stanno in `experiments.csv`: `03_experiments.py` salva
  `results/homogeneity_example.csv` (classi k-anonime con ℓ=1 / più lontane da base rate) e `04` lo legge.
- `ru_map.png` — rischio (asse y) vs utilità (asse x), un punto per configurazione

**Snippet per le slide**: le funzioni chiave (`k_anonymize_fulldomain`,
`l_diversity`, `t_closeness`, una tecnica perturbativa) devono essere brevi e
leggibili, così da poterle mostrare in 2–3 slide.

**Numeri chiave** stampati a fine `03_experiments.py` e salvati in
`results/summary.md`: % record unici prima/dopo, k scelto come compromesso, perdita di AUC
corrispondente, ℓ e t ottenuti.

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

## 5. Definizioni di riferimento
- **Identifier**: identifica univocamente. **Quasi-identifier**: non identifica
  da solo ma è disponibile esternamente e, combinato, permette l'identity linking attack.
- **Equivalence class (q-block)**: record con gli stessi valori dei QI.
- **k-anonymity**: ogni combinazione di valori dei QI compare almeno k volte.
  Si ottiene con generalization (gerarchie, distance vector, generalizzazione
  k-minimale) + suppression degli outlier. Alternativa: **Mondrian**
  (partizionamento top-down: split sulla mediana della dimensione con range
  normalizzato più ampio, finché ogni partizione ha ≥ k record).
- **Attacchi residui**: homogeneity, background knowledge, skewness, similarity.
- **Distinct ℓ-diversity**: ≥ ℓ valori sensibili distinti per classe.
  **Entropy ℓ-diversity**: −Σ p(s)·log p(s) ≥ log ℓ in ogni classe.
- **t-closeness**: in ogni classe EMD(distribuzione nella classe, globale) ≤ t.
  - categorico: ½ · Σ |p_i − q_i|
  - numerico ordinato su m valori: (1/(m−1)) · Σ_i |Σ_{j≤i} (p_j − q_j)|
- **Qualità**: tempo (il problema ottimo è NP-hard), dimensione media delle
  classi, **discernibility** = Σ_t penalty(t), con penalty = |T| se t è
  soppresso, |E_t| altrimenti.
- **Masking**: non perturbative (sampling, global recoding, top/bottom coding,
  generalization, local suppression); perturbative (PRAM, rumore additivo,
  rank swapping, micro-aggregation).

## 6. Gerarchie di generalizzazione (livello 0 = originale)
- `AGE`: esatta → fasce 5 anni → fasce 10 anni → {<30, 30–50, >50} → `*`
- `EDUCATION`: graduate/university/high school/others → {higher, other} → `*`
- `MARRIAGE`: married/single/others → `*`
- `SEX`: male/female → `*`
- `LIMIT_BAL` (solo scenario esteso): fasce 50k → 100k → 200k → `*`

## 7. Struttura
```
src/data.py         caricamento, pulizia, identifier sintetici, sanitize  (FATTO)
src/hierarchies.py  gerarchie e generalize(df, levels)
src/risk.py         uniqueness, record linkage, k, ℓ-diversity, t-closeness
src/utility.py      classificatore (AUC), information loss, discernibility
src/protection.py   masking, k-anonymity full-domain e Mondrian, perturbative
scripts/01_eda.py          statistiche descrittive (stampate) + qi_distributions.png,
                           equivalence_class_sizes_original.png, attribute_classification.csv
scripts/02_baseline.py     rischio e utilità sui dati originali → baseline.csv
scripts/03_experiments.py  k-anonymity, ℓ/t, perturbative → experiments.csv, summary.md
scripts/04_figures.py      risk_vs_k, l_t_vs_k, homogeneity_example, ru_map
                           (legge solo i CSV: si rifanno le figure senza rieseguire gli esperimenti)
main.py             esegue gli script in ordine; opzione per lanciarne uno solo
results/            tabelle CSV + summary.md (sezione 3)
figures/            figure PNG (sezione 3)
```

## 8. Esperimenti
- k ∈ {2, 5, 10, 20, 50}, `max_suppression` = 1%; con QI base e QI estesi.
- Per ogni k: tutte le metriche della riga di `experiments.csv`.
- Perturbative: 2–3 livelli di parametro ciascuna.
- Classificatore: logistic regression, stesso split (`random_state=42`, 70/30
  stratificato) per tutte le versioni; QI generalizzati codificati one-hot.
- Opzionali: Mondrian vs full-domain; libreria `anjana` per verificare i
  risultati della propria implementazione; macrodata (threshold rule, p-percentage).

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