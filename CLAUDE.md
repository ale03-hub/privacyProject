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
- `baseline.csv` — una riga per scenario (QI, QI_EXT): uniqueness, rare_combinations_lt5,
  `det_linkage_rate`, `dist_linkage_rate`, AUC e accuracy (identiche nelle due righe:
  il classificatore non dipende dallo scenario d'attacco)
- `experiments.csv` — formato lungo: una riga per configurazione × scenario d'attacco
  (inclusa `original` e `no_data`, così la R-U map legge un solo file). Colonne:
  `config_id`, `family` (reference / non-perturbative / perturbative / combination),
  `technique`, `params` (stringa leggibile), `scenario` (QI / QI_EXT), `uniqueness`,
  `det_linkage_rate`, `dist_linkage_rate`, `attr_disclosure_rate`, `il_SEX`,
  `il_EDUCATION`, `il_MARRIAGE`, `il_AGE`, `il_LIMIT_BAL`, `il_total`,
  `utility` (= 1 − `il_total`), `auc_full`, `auc_qi`, `suppressed_cells_pct`
  (0 se non applicabile), `time_s` (solo la protezione). Le misure di utilità non
  dipendono dallo scenario: sono calcolate una volta per configurazione e ripetute
  nelle due righe. La local suppression ha varianti separate per QI e QI_EXT, valutate
  solo nel proprio scenario. **È la tabella principale**: R-U map e conclusioni
  derivano da qui.

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
`results/summary.md`: rischio dell'`original` (uniqueness, linkage det e dist, in
entrambi gli scenari); per ogni scenario la configurazione di compromesso, cioè quella
con `utility` più alta tra quelle con `dist_linkage_rate` sotto la soglia `MAX_RISK`
(costante in cima a `03_experiments.py`: 0.05 per QI_EXT, 0.01 per QI), con rischio,
utility, `auc_full`, `auc_qi` e variazione rispetto all'`original`; una tabella
compatta di tutte le configurazioni.

## 4. Dataset
UCI *Default of Credit Card Clients* (id 350): 30.000 clienti di una banca
taiwanese, apr–set 2005. File in `data/raw/` (non versionato).
Il `.xls` ha i nomi reali nella seconda riga → `header=1`.

| Ruolo | Colonne |
|---|---|
| Identifier | `ID` + sintetici `customer_code`, `phone` (simulano lo scenario reale, poi rimossi con `sanitize`) |
| Quasi-identifier | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Sensibili | `default` (principale), `LIMIT_BAL`, `PAY_0`, `PAY_2..PAY_6`, `BILL_AMT1..6`, `PAY_AMT1..6` |

- Scenario d'avversario esteso: chi conosce anche `LIMIT_BAL` **approssimato a fasce
  da 50k** (es. dipendente della banca) → QI estesi `QI + [LIMIT_BAL]`.
- Codici non documentati (`EDUCATION` 0/5/6, `MARRIAGE` 0) accorpati in "others" da `clean()`.
- Target sbilanciato (~22% default): usare **AUC**, non solo accuracy.
- Il dataset è un campione (clienti di una banca), non la popolazione: la
  uniqueness calcolata è **sample uniqueness** e va detto nel report.
- Popolazione / rilascio / esterno: i 30.000 clienti sono la popolazione; il
  **rilascio** è un campione casuale dell'80% (su cui si calcolano uniqueness,
  linkage e AUC), il **dataset esterno** dell'avversario è un campione
  indipendente del 50% della popolazione (≈80% dei suoi record sono nel rilascio,
  gli altri no). Funzione `make_release_and_external` in `src/risk.py`.

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
- **Record linkage** (slide 82-83): l'avversario unisce un record del rilascio a
  un altro dataset tramite attributi comuni, per ampliare ciò che sa sulla persona.
  *Deterministico*: cerca esattamente gli stessi valori degli attributi (limite:
  ignora l'importanza dei singoli attributi). *Distance-based*: ogni record è
  associato al record più simile secondo una misura di similarità f scelta da chi
  analizza, che può pesare gli attributi. Come misura del rischio: dato un
  sottoinsieme di attributi, quanto spesso la strategia riesce a identificare
  correttamente il record. Metrica unica per entrambi:
  `linkage_rate` = n. record esterni linkati alla persona giusta / n. record esterni.
  Regola diversa per i due metodi:
  - *deterministico*: **match univoco**, un record esterno è linkato solo se
    esiste esattamente un record rilasciato con gli stessi QI;
  - *distance-based*: ogni record esterno è **sempre** associato al record
    rilasciato più vicino; in caso di pareggio alla distanza minima se ne sceglie
    uno a caso (rng con seed 42). Non si usa "link solo se il vicino è unico"
    perché l'attaccante si comporterebbe in modo diverso su dati originali (i
    pareggi bloccano il link) e perturbati (i pareggi spariscono e indovina
    sempre), falsando il confronto tra tecniche.
  L'indice di riga è l'identità e serve solo a verificare i link, mai a fare il match.
  - Distanza (scelta progettuale, il corso lascia libera f): numerici (AGE,
    LIMIT_BAL) |Δ|/std dei dati originali, fasce → punto medio; categorici 0 se
    uguali, 1 se diversi; missing (local suppression) 0.5; pesi per attributo
    opzionali (default 1).
- **Cosa conosce l'avversario**: se il rilascio è protetto con tecniche
  deterministiche (generalization, recoding, top/bottom coding) l'avversario
  conosce lo schema e lo applica ai propri dati; se è protetto con tecniche
  perturbative (e con local suppression) ha solo i valori originali. Lo schema
  è `{colonna: livello di gerarchia}` per recoding/generalization, oppure una
  funzione per il top/bottom coding. In ogni caso il suo `LIMIT_BAL` è a
  fasce da 50k, e il rilascio viene portato nello stesso dominio solo per il
  confronto (il dataset rilasciato non si modifica).
- **Attribute disclosure rate** (misura nostra, dalla definizione di slide 42):
  % di record che stanno in un gruppo di QI con ≥ 2 record e `default` costante
  (i missing contano come valore a sé). È calcolata **nel dominio dell'attaccante**:
  il rilascio viene portato nello stesso dominio del dataset esterno con lo stesso
  helper di allineamento del linkage (`LIMIT_BAL` a fasce da 50k nello scenario
  esteso, schema della protezione deterministica), solo per il calcolo. Ha senso
  **solo per original e tecniche non perturbative**: con quelle perturbative i
  gruppi del rilascio non corrispondono più a persone reali (il rumore rende ogni
  record unico e la misura crolla a ~0; la micro-aggregazione crea gruppi
  artificiali e la fa salire), quindi in `experiments.csv` vale NaN per le
  configurazioni perturbative e per quelle con una parte perturbativa.
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
  Come è implementato (`src/utility.py`, slide 85-86):
  - continuo: sulla vista numerica dei valori protetti (`to_numeric_view` in
    `src/numeric.py`: fasce → punto medio, `<x`/`>x` → x, missing → NaN); MSE e MAE
    sui valori standardizzati con la std originale, variazione relativa (assoluta)
    di media e varianza; le righe con missing sono escluse e se ne riporta la quota;
  - categorico: `changed_share` (missing = cambiato), `table_distance` = ½ Σ|p−q|
    tra le frequenze congiunte (colonna, `default`) prima e dopo, `cond_entropy` =
    H(originale | protetto) in bit. I valori generalizzati si confrontano come
    stringhe: per generalization/recoding `changed_share` e `table_distance`
    saturano (≈1) e l'unica misura informativa è l'entropia condizionata.
  Scelte di implementazione del masking (`src/protection.py`): local suppression
  azzera un QI alla volta (quello con più valori distinti: prima `LIMIT_BAL`/`AGE`) nei
  record in combinazioni rare e restituisce anche la % di celle QI soppresse;
  rank swapping con `p` in percentuale di n; rumore additivo con `AGE` arrotondata
  e clip al minimo osservato; micro-aggregation 1D sulla prima componente
  principale (versione semplificata del clustering per massima similarità).
- **Information loss totale** (`information_loss_total` in `src/utility.py`): misura
  unica, calcolata sempre sulle cinque colonne `SEX, EDUCATION, MARRIAGE, AGE,
  LIMIT_BAL` qualunque sia lo scenario d'attacco, così le configurazioni sono
  confrontabili. Per colonna una perdita in [0, 1] (0 = dati intatti, 1 = "no data"):
  categoriche H(originale | protetto) / H(originale) in bit, valori confrontati come
  stringhe e missing come categoria a sé; numeriche MAE(originale, protetto) /
  MAE(originale, media dell'originale) sulla vista numerica del protetto, missing
  sostituiti dalla media originale, clip a 1. `il_total` = media delle cinque.
- **R-U map**: utilità sull'asse x, rischio (`dist_linkage_rate`) sull'asse y,
  soglia di massimo rischio tollerabile. Il corso definisce l'utilità come 1/information
  loss; usiamo **utility = 1 − il_total** perché 1/IL diverge per i dati originali.
  L'AUC non è usata come asse x perché non separa le tecniche (`auc_full` resta tra
  0.722 e 0.725; `auc_qi` varia di ~0.015): è riportata come colonna a parte.

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
src/numeric.py      to_numeric_view, to_labels: lettura numerica/categorica di colonne protette  (FATTO)
src/hierarchies.py  gerarchie e generalize(df, levels)  (FATTO)
src/risk.py         sample uniqueness, make_release_and_external, record linkage
                    (deterministico e distance-based, con allineamento del dominio),
                    attribute disclosure rate  (FATTO)
src/utility.py      classificatore (AUC), information loss continuo e categorico  (FATTO)
src/protection.py   masking non perturbativo e perturbativo  (FATTO)
scripts/01_eda.py          statistiche descrittive (stampate) + qi_distributions.png,
                           qi_combination_sizes_original.png, attribute_classification.csv
scripts/02_baseline.py     rischio e utilità sui dati originali → baseline.csv
scripts/03_experiments.py  tutte le configurazioni → experiments.csv, summary.md  (FATTO)
scripts/04_figures.py      risk_by_technique, utility_by_technique, ru_map
                           (legge solo i CSV: si rifanno le figure senza rieseguire gli esperimenti)
main.py             esegue gli script in ordine; opzione per lanciarne uno solo
results/            tabelle CSV + summary.md (sezione 3)
figures/            figure PNG (sezione 3)
```

## 8. Esperimenti
- Rilascio = campione casuale dell'80% della popolazione (seed 42); dataset
  esterno dell'avversario = campione indipendente del 50% della popolazione, con i
  QI originali e `LIMIT_BAL` a fasce da 50k. Il linkage (deterministico e
  distance-based; metrica `linkage_rate`, regole diverse: match univoco vs
  più vicino con pareggi a caso, vedi sezione 5) si fa contro il rilascio protetto. Tecniche deterministiche →
  l'avversario applica lo schema ai suoi dati; perturbative → ha solo i valori
  originali (vedi sezione 5).
- Ogni configurazione si applica **una volta** al rilascio e si valuta in entrambi gli
  scenari (QI e QI_EXT); eccezione: la local suppression dipende dai QI, quindi ha
  varianti separate per QI e QI_EXT, valutate solo nel proprio scenario. Nelle
  combinazioni con una parte deterministica e una perturbativa (o local suppression)
  l'avversario applica solo la parte deterministica. La griglia è una lista di dict in
  cima a `scripts/03_experiments.py` (31 configurazioni, tra cui le due di riferimento):
  - riferimento: `original`, `no_data` (QI e `LIMIT_BAL` tutti a missing)
  - non perturbative:
    - global recoding di `AGE`: fasce 5 / 10 / 3 fasce (livello 3 della gerarchia)
    - top/bottom coding: (`AGE` >60, `LIMIT_BAL` >500k) e (`AGE` <25 e >55, `LIMIT_BAL` >300k)
    - generalization: `EDUCATION`, `MARRIAGE`, entrambe (livello 1)
    - local suppression con soglia ∈ {2, 5, 10}, varianti QI e QI_EXT
  - perturbative (sempre su entrambe le colonne della tecnica):
    - PRAM su `EDUCATION` + `MARRIAGE`: prob. di mantenere il valore ∈ {0.9, 0.7, 0.5}
    - rumore additivo su `AGE` + `LIMIT_BAL`: σ ∈ {0.05, 0.1, 0.2} · std
    - rank swapping su `AGE` + `LIMIT_BAL`: p ∈ {5, 10, 20}%
    - micro-aggregation su `AGE` + `LIMIT_BAL`: cluster di dimensione ∈ {3, 5, 10}
  - combinazioni: recoding `AGE` 10 + generalization `EDUCATION` e `MARRIAGE` + local
    suppression soglia 5 (QI e QI_EXT); rumore 0.1 su `AGE` + `LIMIT_BAL` + PRAM 0.9
    su `EDUCATION` + `MARRIAGE`.
- Classificatore: logistic regression, stesso split (`random_state=42`, 70/30
  stratificato, calcolato una volta sul rilascio originale) per tutte le versioni.
  La codifica dipende solo dal tipo di colonna dopo la protezione: `SEX`,
  `EDUCATION`, `MARRIAGE` e ogni colonna non numerica (generalizzata) o con
  missing → one-hot (`handle_unknown="ignore"`, missing come categoria a sé); le
  altre numeriche (`AGE`, `LIMIT_BAL`, `PAY_*`, `BILL_AMT*`, `PAY_AMT*`) →
  `StandardScaler`. Quindi l'AUC non cambia con lo scenario d'attacco (QI vs QI_EXT).
  Si riportano due AUC: `auc_full` (tutte le feature) e `auc_qi` (solo `SEX, EDUCATION,
  MARRIAGE, AGE, LIMIT_BAL`, parametro `features` di `evaluate_classifier`). Nota:
  proteggere solo i QI incide poco sull'AUC (sono poco predittivi, il lavoro lo
  fanno `PAY_*`); perturbare `LIMIT_BAL` incide di più.
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