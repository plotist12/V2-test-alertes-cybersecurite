# Résumeur RSS (Google Alerts) — 100% local et gratuit

Ce projet récupère un flux RSS **Google Alerts**, extrait le texte des articles, et génère **un fichier Markdown par jour** contenant les **résumés** de tous les articles parus ce jour-là.  
Aucune clé API payante n'est nécessaire : la synthèse est **locale** (LexRank via `sumy`).

## 🚀 Démarrage (local)

1. **Installez Python 3.10+**.
2. Ouvrez un terminal dans le dossier du projet, puis :
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. Lancez le script pour **aujourd'hui** (heure Europe/Paris) :
   ```bash
   python main.py
   ```
   Ou pour une date précise (format `YYYY-MM-DD`) :
   ```bash
   python main.py --date 2025-09-17
   ```

- Les résumés sont écrits dans `output/AAAA-MM-JJ.md`.
- La page `index.html` est (re)générée pour lister les jours déjà disponibles.

> Par défaut, le flux utilisé est celui fourni dans votre demande :
>
> `https://www.google.fr/alerts/feeds/06828307252155608763/17948382890845885017`
>
> Vous pouvez en mettre un autre avec `--feed URL_DU_FLUX`.

## ❓ Comment ça marche

- **Extraction** : on télécharge chaque lien du flux et on récupère le texte (via `BeautifulSoup`, heuristiques).
- **Résumé** : on applique **LexRank** (lib `sumy`), en français si détecté, sinon en anglais. Aucun service externe requis.
- **Groupement par jour** : on utilise la date de publication détectée dans le flux, convertie en **Europe/Paris**. On génère **un Markdown par jour**.
- **Index** : `index.html` liste simplement les fichiers du dossier `output/`.

## ⚙️ Options utiles

```bash
python main.py \
  --feed "URL_DU_FLUX" \
  --date 2025-09-17 \
  --out output \
  --sentences 5 \
  --skip-index
```

## 🌐 Publication automatique sur GitHub Pages (1 fois / jour)

1. Créez un dépôt public sur GitHub (ex: `rss-summarizer`), ajoutez-y ces fichiers, puis poussez.
2. Activez **GitHub Pages** sur la branche `gh-pages` (ou `main` si vous préférez).  
3. L'action ci-dessous (dans `.github/workflows/daily.yml`) va :
   - S'exécuter tous les jours à 07:00 (heure de Paris),
   - Installer Python + dépendances,
   - Lancer `python main.py` (pour la date du jour),
   - **Commit & push** les changements (nouveaux fichiers `output/*.md` + `index.html`).

> Aucun secret n'est requis : l'action utilise le jeton automatique `GITHUB_TOKEN` avec les permissions d'écriture.

## 🧪 Test rapide

Après avoir lancé `python main.py`, ouvrez `index.html` dans votre navigateur et cliquez sur une date.

## 📝 Licence

MIT — faites-en ce que vous voulez. Contributions bienvenues !


# commande a lancer pour le résumé avec ia (petit modèle)
python main.py --summarizer hf --hf-model csebuetnlp/mT5_small_m2o_xlsum --points 6

2. Comment activer ton venv dans VS Code (Windows)

Ton venv est probablement dans .venv\.
Pour l’activer dans ton terminal PowerShell de VS Code, tape :

.\.venv\Scripts\Activate.ps1


# pour se placer dans le bon dossier et lancer le petit programme

# Depuis le dossier parent
cd "rss_summarizer"

# Vérifie que main.py est bien là
dir main.py

# Lance avec le petit modèle mT5
python .\main.py --summarizer hf --hf-model csebuetnlp/mT5_small_m2o_xlsum --points 6
