"""Download a small, explicitly matched poster collection for the local demo.

Run manually; the server never needs internet access. Image provenance is
saved in static/posters/sources.json. Existing downloads are reused.
"""

import json
import ssl
import time
from html import escape
from urllib.error import HTTPError
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


POSTERS = Path(__file__).resolve().parent / "static" / "posters"
TITLES = {
    318: "The Shawshank Redemption", 356: "Forrest Gump",
    296: "Pulp Fiction", 2571: "The Matrix",
    593: "The Silence of the Lambs (film)", 260: "Star Wars (film)",
    2959: "Fight Club", 480: "Jurassic Park (film)",
    527: "Schindler's List", 4993: "The Lord of the Rings: The Fellowship of the Ring",
    1196: "The Empire Strikes Back", 110: "Braveheart",
    1: "Toy Story", 589: "Terminator 2: Judgment Day",
    50: "The Usual Suspects", 1210: "Return of the Jedi",
    5952: "The Lord of the Rings: The Two Towers",
    7153: "The Lord of the Rings: The Return of the King",
    1198: "Raiders of the Lost Ark", 858: "The Godfather",
    2858: "American Beauty (1999 film)", 47: "Seven (1995 film)",
    1270: "Back to the Future", 58559: "The Dark Knight",
    2028: "Saving Private Ryan", 608: "Fargo (1996 film)",
    79132: "Inception", 3578: "Gladiator (2000 film)",
    780: "Independence Day (1996 film)", 457: "The Fugitive (1993 film)",
    2762: "The Sixth Sense", 150: "Apollo 13 (film)",
    32: "12 Monkeys", 4306: "Shrek", 4226: "Memento (film)",
    364: "The Lion King (1994 film)", 1704: "Good Will Hunting",
    588: "Aladdin (1992 Disney film)", 592: "Batman (1989 film)",
    6539: "Pirates of the Caribbean: The Curse of the Black Pearl",
    3114: "Toy Story 2", 78499: "Toy Story 3", 60069: "WALL-E",
    201588: "Toy Story 4",
    68954: "Up (2009 film)", 8961: "The Incredibles",
    6377: "Finding Nemo", 231: "Dumb and Dumber",
    81834: "Harry Potter and the Deathly Hallows – Part 1",
}


def main():
    context = ssl.create_default_context()
    try:
        from pip._vendor import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    def fetch(url):
        request = Request(url, headers={
            "User-Agent": "Movie Recommender-MovieLens-LocalDemo/1.0 (educational poster thumbnails)"})
        with urlopen(request, context=context, timeout=30) as response:
            return response.read()

    POSTERS.mkdir(parents=True, exist_ok=True)
    manifest = POSTERS / "sources.json"
    sources = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    query = urlencode({"action": "query", "format": "json", "formatversion": 2,
                       "prop": "pageimages", "titles": "|".join(TITLES.values()),
                       "pithumbsize": 400, "pilicense": "any", "pilimit": 50,
                       "redirects": 1})
    result = json.loads(fetch("https://en.wikipedia.org/w/api.php?" + query))["query"]
    pages = {page["title"]: page for page in result["pages"]}
    aliases = {r["from"]: r["to"] for key in ("normalized", "redirects")
               for r in result.get(key, [])}
    for movie_id, title in TITLES.items():
        if str(movie_id) in sources and (POSTERS / sources[str(movie_id)]["file"]).exists():
            continue
        resolved = title
        for _ in range(5):
            resolved = aliases.get(resolved, resolved)
        page = pages.get(resolved, {})
        thumb = page.get("thumbnail", {})
        url = thumb.get("source")
        if not url or thumb.get("width", 1) >= thumb.get("height", 0):
            print(f"No portrait poster for {movie_id}; keeping the fallback.")
            continue
        suffix = Path(urlsplit(url).path).suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            time.sleep(2)
            data = fetch(url)
        except HTTPError as error:
            if error.code == 429:
                delay = min(60, max(10, int(error.headers.get("Retry-After", "10"))))
                print(f"Poster service busy; waiting {delay}s.", flush=True)
                time.sleep(delay)
                try:
                    data = fetch(url)
                except OSError:
                    print("Poster service remains busy. Rerun later to resume.")
                    break
            else:
                print(f"Poster {movie_id} unavailable: HTTP {error.code}")
                continue
        except OSError as error:
            print(f"Poster {movie_id} unavailable: {error}")
            continue
        filename = f"{movie_id}{suffix}"
        (POSTERS / filename).write_bytes(data)
        sources[str(movie_id)] = {
            "file": filename, "title": title, "image_url": url,
            "article_url": "https://en.wikipedia.org/wiki/" + quote(resolved.replace(" ", "_")),
            "file_page": "https://en.wikipedia.org/wiki/File:" + quote(page["pageimage"]),
        }
        manifest.write_text(json.dumps(sources, indent=2), encoding="utf-8")
        print(f"Saved poster {movie_id}", flush=True)
    credits = ['<!doctype html><html lang="en"><meta charset="utf-8">',
               '<meta name="viewport" content="width=device-width,initial-scale=1">',
               '<title>Movie Recommender · Image credits</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:850px;margin:40px auto;padding:20px;background:#111517;color:#eee}a{color:#b9f28b}li{margin:12px 0}</style>',
               '<a href="/">← Back to Movie Recommender</a><h1>Image credits</h1>',
               '<p>Movie poster thumbnails from Wikipedia. Artwork belongs to the respective rights holders. Source pages provide image details.</p><ul>']
    for item in sources.values():
        credits.append(f'<li>{escape(item["title"])} — <a href="{escape(item["file_page"])}">Image source</a></li>')
    credits.append('</ul></html>')
    (POSTERS.parent / "poster-credits.html").write_text('\n'.join(credits), encoding="utf-8")
    print(f"{len(sources)} local posters ready.")


if __name__ == "__main__":
    main()
