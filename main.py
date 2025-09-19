#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py (v5 — bullets de qualité via IA locale)

- Déroule les liens Google Alerts (google.com/url?url=...)
- Extraction article: Trafilatura (fallback BeautifulSoup) + fallback RSS
- Résumé IA:
    * 'hf' (par défaut): modèle mT5 XLSum (Hugging Face), avec map-reduce
    * 'lexrank': extractif léger (fallback automatique)
- Sortie: bullet points concises.

Dépendances de base: feedparser, requests, bs4, lxml (<5.2), trafilatura, sumy, langdetect
Option IA: transformers, sentencepiece, accelerate, torch (CPU)
"""

import argparse
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from email.utils import parsedate_to_datetime


import feedparser
import requests
from bs4 import BeautifulSoup

# -------- Extraction robuste --------
try:
    import trafilatura
except Exception:
    trafilatura = None

# -------- Résumé extractif (fallback) --------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Lang detect (optionnel)
try:
    from langdetect import detect
except Exception:
    detect = None

DEFAULT_FEED = "https://www.google.fr/alerts/feeds/06828307252155608763/17948382890845885017"
DEFAULT_LANG = "french"
PARIS_TZ = "Europe/Paris"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ------------- utilitaires -------------
def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()
    return soup.get_text(" ", strip=True)


def fetch_url(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124 Safari/537.36"),
        "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text


def unwrap_google_redirect(url: str) -> str:
    try:
        u = urlparse(url)
        if "google." in u.netloc.lower() and u.path.startswith("/url"):
            qs = parse_qs(u.query)
            for key in ("url", "q", "u"):
                if key in qs and qs[key]:
                    return unquote(qs[key][0])
        return url
    except Exception:
        return url


def extract_main_text(html: str, url: str = "") -> str:
    # 1) Trafilatura si dispo
    if trafilatura:
        try:
            txt = trafilatura.extract(
                html, url=url or None,
                include_comments=False, include_tables=False, favor_recall=True
            )
            if txt and len(txt.split()) > 40:
                return txt
        except Exception:
            pass
    # 2) Fallback BeautifulSoup
    soup = BeautifulSoup(html or "", "lxml")
    for bad in soup(["script", "style", "noscript", "header", "footer", "form", "iframe", "svg"]):
        bad.decompose()
    article = soup.find("article")
    container = article or soup.find("main") or soup.find("div", {"id": "content"}) or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in container.find_all(["p", "li"])]
    text = "\n".join([t for t in paras if len(t.split()) > 4])
    if not text or len(text.split()) < 50:
        title = soup.title.get_text(strip=True) if soup.title else ""
        desc = ""
        m = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if m and m.get("content"):
            desc = m["content"]
        text = (title + "\n\n" + desc).strip()
    return text


def sentence_splits(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?…])\s+", (text or "").strip())
    return [s.strip() for s in parts if len(s.split()) >= 5]


def chunk_by_words(text: str, max_words: int = 450) -> list[str]:
    """Découpe proprement par phrases pour éviter de tronquer violemment."""
    sents = sentence_splits(text)
    chunks, buf, count = [], [], 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and buf:
            chunks.append(" ".join(buf))
            buf, count = [s], w
        else:
            buf.append(s); count += w
    if buf:
        chunks.append(" ".join(buf))
    return chunks if chunks else [text]


# ------------- Résumé IA (HuggingFace) -------------
class HFXlsum:
    _pipe = None
    _model = None

    @classmethod
    def ensure(cls, model_name: str) -> bool:
        if cls._pipe and cls._model == model_name:
            return True
        try:
            from transformers import pipeline
            cls._pipe = pipeline(
                "summarization",
                model=model_name,      # mT5 XLSum
                device=-1,             # CPU
            )
            cls._model = model_name
            logging.info(f"HF chargé: {model_name}")
            return True
        except Exception as e:
            logging.warning(f"HF indisponible ({model_name}): {e}")
            cls._pipe = None
            return False

    @classmethod
    def summarize_to_bullets(cls, text: str, n_points: int, model_name: str) -> list[str]:
        if not text or not cls.ensure(model_name):
            return []
        # Map: résumer chaque chunk
        chunks = chunk_by_words(text, max_words=450)
        partials = []
        for ch in chunks:
            try:
                out = cls._pipe(ch, max_length=160, min_length=70, truncation=True)
                partials.append(out[0]["summary_text"].strip())
            except Exception as e:
                logging.warning(f"sous-résumé HF KO: {e}")
        if not partials:
            return []
        # Reduce: re-résumer l'ensemble
        combined = " ".join(partials)
        try:
            final = cls._pipe(combined, max_length=180, min_length=90, truncation=True)
            summary = final[0]["summary_text"].strip()
        except Exception as e:
            logging.warning(f"résumé final HF KO: {e}")
            summary = combined

        # Transformer en bullets: découpe + nettoyage
        sents = sentence_splits(summary)
        # Fusionner les phrases trop courtes avec la suivante
        bullets = []
        i = 0
        while i < len(sents):
            s = sents[i]
            if len(s.split()) < 6 and i + 1 < len(sents):
                s = (s + " " + sents[i+1]).strip()
                i += 1
            bullets.append(s.strip())
            i += 1
        bullets = [b.strip(" -•\u2022") for b in bullets if len(b.split()) >= 6]
        return bullets[:n_points] if bullets else []


def lexrank_summary(text: str, max_sentences: int = 6, language_hint: str = DEFAULT_LANG) -> str:
    words = (text or "").split()
    if len(words) < 40:
        return " ".join(words)
    lang = language_hint
    if detect:
        try:
            lc = detect(" ".join(words[:2000]))
            mapping = {"fr": "french", "en": "english", "es": "spanish",
                       "de": "german", "it": "italian", "pt": "portuguese"}
            lang = mapping.get(lc, language_hint)
        except Exception:
            pass
    try:
        parser = PlaintextParser.from_string(" ".join(words), Tokenizer(lang))
        summarizer = LexRankSummarizer(Stemmer(lang))
        summarizer.stop_words = get_stop_words(lang)
        sents = summarizer(parser.document, max_sentences)
        out = " ".join(str(s) for s in sents).strip()
        return out if out else " ".join(words[:200])
    except Exception:
        return " ".join(words[:200])


def bullets_from_text(text: str, n: int) -> list[str]:
    sents = sentence_splits(text)
    if not sents:
        return [text.strip()] if text else []
    # Garder les 2–3 plus longues (souvent plus informatives) + le reste jusqu'à n
    sents_sorted = sorted(sents, key=lambda s: -len(s))
    pick = set(sents_sorted[: max(2, n//2)])
    result = []
    for s in sents:
        if s in pick or len(result) < n:
            result.append(s)
        if len(result) >= n:
            break
    return result[:n]


# ------------- RSS helpers -------------
def get_entry_datetime(entry) -> datetime:
    """
    Retourne la date/heure UTC de l'entrée en essayant d'abord les champs texte
    (published/updated), puis les champs *_parsed. Jamais 'now()' sauf en dernier recours.
    """
    # 1) Champs texte (RFC 2822 / ISO) -> parsés proprement
    for attr in ("published", "updated", "dc_date", "created"):
        val = getattr(entry, attr, None)
        if val:
            try:
                dt = parsedate_to_datetime(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass

    # 2) Tuples *_parsed (si fournis par feedparser)
    for key in ("published_parsed", "updated_parsed"):
        dt_tuple = getattr(entry, key, None)
        if dt_tuple:
            return datetime(*dt_tuple[:6], tzinfo=timezone.utc)

    # 3) Dernier recours : ne **pas** fausser le filtrage → remonter "1970-01-01"
    # (ainsi, --date all le récupérera, mais un filtre précis ne le sélectionnera pas par erreur)
    return datetime(1970, 1, 1, tzinfo=timezone.utc)



def group_by_paris_date(dt_utc: datetime) -> str:
    import pytz
    paris = pytz.timezone(PARIS_TZ)
    return dt_utc.astimezone(paris).strftime("%Y-%m-%d")


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def get_feed_text(entry) -> str:
    try:
        if hasattr(entry, "content") and entry.content:
            c0 = entry.content[0]
            val = c0["value"] if isinstance(c0, dict) and "value" in c0 else getattr(c0, "value", "")
            if val:
                txt = clean_html_to_text(val)
                if len(txt.split()) > 10:
                    return txt
    except Exception:
        pass
    desc = getattr(entry, "summary", "") or getattr(entry, "description", "")
    desc = clean_html_to_text(desc)
    if len(desc.split()) > 10:
        return desc
    return clean_html_to_text(getattr(entry, "title", "") or "")


# ------------- cœur -------------
def process_feed(feed_url: str, only_date: str | None, out_dir: str, n_points: int,
                 prefer_source: str, summarizer: str, hf_model: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Lecture du flux: {feed_url}")
    fp = feedparser.parse(feed_url)

    items_by_date: dict[str, list[dict]] = {}

    for entry in fp.entries:
        raw_link = getattr(entry, "link", None) or getattr(entry, "id", None)
        title = clean_html_to_text(getattr(entry, "title", "Sans titre"))
        if not raw_link:
            logging.warning(f"Entrée sans lien ignorée: {title}")
            continue

        link = unwrap_google_redirect(raw_link)
        published_dt = get_entry_datetime(entry)
        day_key = group_by_paris_date(published_dt)
        if only_date and day_key != only_date:
            continue

        # Texte candidat
        feed_text = get_feed_text(entry)
        article_text = ""
        if prefer_source == "rss":
            article_text = feed_text
        else:
            try:
                html = fetch_url(link)
                article_text = extract_main_text(html, url=link)
            except Exception as e:
                logging.warning(f"Extraction KO {link}: {e}")
            if (not article_text) or len(article_text.split()) < 60:
                article_text = feed_text
        if not article_text.strip():
            article_text = f"{title} — {link}"

        # Résumé → bullets
        bullets = []
        if summarizer in ("hf", "auto"):
            bullets = HFXlsum.summarize_to_bullets(article_text, n_points=n_points, model_name=hf_model)

        if not bullets:  # fallback
            compact = lexrank_summary(article_text, max_sentences=min(8, n_points + 3), language_hint=DEFAULT_LANG)
            bullets = bullets_from_text(compact, n_points)

        items_by_date.setdefault(day_key, []).append({
            "title": title,
            "url": link,
            "domain": domain_from_url(link),
            "published_iso": published_dt.isoformat(),
            "bullets": bullets,
        })

    # Écriture Markdown par jour
    for day, items in sorted(items_by_date.items()):
        md_path = Path(out_dir) / f"{day}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# Résumés des articles — {day}\n\n")
            for it in items:
                f.write(f"## {it['title']}\n")
                f.write(f"[{it['domain']}]({it['url']}) — Publié: {it['published_iso']}\n\n")
                for b in (it["bullets"] or ["(résumé indisponible)"]):
                    f.write(f"- {b}\n")
                f.write("\n---\n\n")
        logging.info(f"Écrit: {md_path} ({len(items)} articles)")

    return items_by_date


def build_index_html(out_dir: str = "output", site_title: str = "Résumés du flux Google Alerts"):
    files = sorted(Path(out_dir).glob("*.md"), reverse=True)
    li = [f'<li><a href="output/{p.name}">{p.stem}</a></li>' for p in files]
    html = f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{site_title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.8rem; }} ul {{ line-height: 1.9; }}
footer {{ margin-top: 3rem; font-size:.9rem; color:#666; }}
.hint {{ background:#f3f4f6; padding:.75rem 1rem; border-radius:.5rem; }}
</style></head>
<body>
<h1>{site_title}</h1>
<p class="hint">Cliquez un jour pour voir les résumés (fichiers Markdown dans <code>output/</code>).</p>
<ul>{"".join(li) if li else "<li>Aucun fichier encore. Lancez le script.</li>"}</ul>
<footer>Généré par <code>main.py</code> — modèle mT5 XLSum si disponible, sinon fallback LexRank.</footer>
</body></html>"""
    Path("index.html").write_text(html, encoding="utf-8")
    return "index.html"


def main():
    parser = argparse.ArgumentParser(description="Résumés bullet points (IA locale) pour un flux Google Alerts.")
    parser.add_argument("--feed", default=DEFAULT_FEED, help="URL du flux RSS.")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (Europe/Paris). Si 'all': toutes les dates du flux.")
    parser.add_argument("--out", default="output", help="Dossier de sortie.")
    parser.add_argument("--points", type=int, default=5, help="Nb de bullets par article.")
    parser.add_argument("--prefer", choices=["article", "rss"], default="article",
                        help="article: extraire; rss: résumer le texte du flux.")
    parser.add_argument("--summarizer", choices=["hf", "lexrank", "auto"], default="hf",
                        help="hf=HuggingFace; lexrank=extractif; auto=HF si dispo sinon fallback.")
    parser.add_argument("--hf-model", default="csebuetnlp/mT5_small_m2o_xlsum",
                        help="ex: csebuetnlp/mT5_small_m2o_xlsum (rapide) ou csebuetnlp/mT5_multilingual_XLSum (qualité+).")
    parser.add_argument("--skip-index", action="store_true", help="Ne pas régénérer index.html.")
    args = parser.parse_args()

    # date
    if args.date is None:
        import pytz
        paris = pytz.timezone(PARIS_TZ)
        date_filter = datetime.now(tz=paris).strftime("%Y-%m-%d")
    elif isinstance(args.date, str) and args.date.lower() == "all":
        date_filter = None
    else:
        date_filter = args.date

    items = process_feed(
        args.feed,
        only_date=date_filter,
        out_dir=args.out,
        n_points=args.points,
        prefer_source=args.prefer,
        summarizer=args.summarizer,
        hf_model=args.hf_model,
    )
    if not args.skip_index:
        build_index_html(args.out)
    if not items:
        logging.info("Aucun article pour cette date, mais index.html mis à jour.")


if __name__ == "__main__":
    main()
