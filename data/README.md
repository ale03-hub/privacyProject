# Dati

Il dataset non è versionato. Due modi per ottenerlo:

1. **Automatico** (consigliato): `src/data.py` usa il pacchetto `ucimlrepo`
   (`fetch_ucirepo(id=350)`) e salva una copia in `data/raw/credit_default.csv`.
2. **Manuale**: scarica lo zip dalla pagina UCI ed estrai
   `default of credit card clients.xls` in `data/raw/`.

Nota: nel file .xls la prima riga contiene nomi generici (X1...X23, Y),
la seconda i nomi reali, quindi si legge con `header=1`.
