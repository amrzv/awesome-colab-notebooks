from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from json import load
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from cv2 import IMREAD_GRAYSCALE, THRESH_BINARY, imdecode, threshold
from google.cloud import bigquery
from matplotlib import font_manager
from numpy import any, asarray, mean, median
from pypistats import overall, recent
from tqdm import tqdm
from wordcloud import WordCloud

BADGES = {image.stem for image in Path("images").glob("*.svg")}
TOP_K = 20


def read_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return load(f)


def load_projects() -> list[dict]:
    return read_json("data/data.json")


def colab_url(url: str) -> str:
    return f"[![Open In Colab](images/colab.svg)]({url})"


def doi_url(url: str) -> str:
    doi = url.split("org/")[1]
    return f"[![](https://api.juleskreuer.eu/citation-badge.php?doi={doi})]({url})"


def git_url(url: str) -> str:
    repo = "/".join(url.split("com/")[1].split("/")[:2])
    return f"[![](https://img.shields.io/github/stars/{repo}?style=social)]({url})"


def pypi_url(package: str, period: str = "dm") -> str:
    return f"[![](https://img.shields.io/pypi/{period}/{package}?style=flat&logo=pypi&label=%E2%80%8D&labelColor=f7f7f4&color=006dad)](https://pypi.org/project/{package}/)"


def infer_link_key(url: str, name_hint: str | None = None) -> str | None:
    if isinstance(name_hint, str) and name_hint.strip():
        return name_hint.strip().casefold()

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host.startswith("www."):
        host = host[4:]

    candidates: list[str] = []
    if host == "github.com":
        candidates.append("git")
    if host == "doi.org":
        candidates.append("doi")
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}:
        candidates.append("yt")
    if host == "arxiv.org":
        candidates.append("arxiv")
    if host == "huggingface.co":
        candidates.append("hf")
    if host == "kaggle.com":
        candidates.append("kaggle")
    if host.endswith("medium.com"):
        candidates.append("medium")
    if host in {"meta.com", "ai.meta.com"}:
        candidates.append("meta")
    if host in {"discord.gg", "discord.com"}:
        candidates.append("discord")
    if host in {"hub.docker.com", "docker.com"}:
        candidates.append("docker")
    if host.endswith("readthedocs.io") or host.startswith("docs."):
        candidates.append("docs")
    if host == "pytorch.org":
        candidates.append("pt")
    if host == "paperswithcode.com":
        candidates.append("pwc")
    if host == "pypi.org":
        candidates.append("pypi")
    if host.endswith("reddit.com"):
        candidates.append("reddit")
    if host.endswith("slack.com"):
        candidates.append("slack")
    if host.endswith("tensorflow.org"):
        candidates.append("tf")
    if host in {"twitter.com", "x.com"}:
        candidates.append("twitter")
    if host.endswith("wikipedia.org"):
        candidates.append("wiki")
    if host.endswith("deepmind.com"):
        candidates.append("deepmind")
    if host.endswith("neurips.cc") or host.endswith("nips.cc"):
        candidates.append("neurips")
    if path.endswith(".pdf"):
        candidates.append("pdf")

    always_keep = {"doi", "git"}
    for key in candidates:
        if key in always_keep or key in BADGES:
            return key
    return None


def normalize_authors(project: dict) -> list[tuple[str, str]]:
    raw_authors = project.get("authors", project.get("author", []))
    authors: list[tuple[str, str]] = []
    for author in raw_authors:
        if isinstance(author, dict):
            name = author.get("name")
            url = author.get("url")
            if isinstance(name, str) and isinstance(url, str):
                authors.append((name, url))
        elif isinstance(author, (list, tuple)) and len(author) >= 2:
            if isinstance(author[0], str) and isinstance(author[1], str):
                authors.append((author[0], author[1]))
    return authors


def get_link_tuples(project: dict) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []
    for link in project.get("links", []):
        if isinstance(link, str):
            out.append((infer_link_key(link), link))
        elif isinstance(link, dict) and isinstance(link.get("url"), str):
            out.append((infer_link_key(link["url"], link.get("name")), link["url"]))
    return out


def get_first_link(project: dict, key: str) -> str | None:
    for link_key, link_url in get_link_tuples(project):
        if link_key == key:
            return link_url
    return None


def get_git_name_stars(git_link: tuple[str, int]) -> tuple[str, int]:
    url, stars = git_link
    idx = url.index("/", 19) + 1
    idx = url.find("/", idx)
    name = url[:idx] if idx != -1 else url
    return name, stars


def parse_link(link_tuple: tuple[str | None, str], height: int = 20) -> str:
    name, url = link_tuple
    if name in BADGES:
        return f'[<img src="images/{name}.svg" alt="{name}" height={height}/>]({url})'
    label = name if isinstance(name, str) else url
    return f"[{label}]({url})"


def parse_authors(authors: list[tuple[str, str]], num_of_visible: int) -> str:
    if len(authors) == 0:
        return ""
    if len(authors) == 1:
        return "[{}]({})".format(*authors[0])
    if len(authors) <= num_of_visible + 1:
        return "<ul>" + " ".join(f"<li>[{author}]({link})</li>" for author, link in authors[: num_of_visible + 1]) + "</ul>"
    return (
        "<ul>"
        + " ".join(f"<li>[{author}]({link})</li>" for author, link in authors[:num_of_visible])
        + "<details><summary>others</summary>"
        + " ".join(f"<li>[{author}]({link})</li>" for author, link in authors[num_of_visible:])
        + "</ul></details>"
    )


def parse_links(links: list[tuple[str | None, str]]) -> str:
    if len(links) == 0:
        return ""

    dct: dict[str | None, list[str]] = defaultdict(list)
    for name, url in links:
        dct[name].append(url)

    line = ""
    if "doi" in dct:
        line += doi_url(dct["doi"][0]) + " "
        dct.pop("doi")
    if "git" in dct:
        line += git_url(dct["git"][0]) + " "
        if len(dct["git"]) == 1:
            dct.pop("git")
        else:
            dct["git"].pop(0)
    if len(dct) == 0:
        return line

    ordered_names = sorted((name for name in dct if name is not None), key=lambda n: n.casefold()) + ([None] if None in dct else [])
    return line + "<ul>" + "".join(
        (
            "<li>" + ", ".join(f"[{idx}]({url})" for idx, url in enumerate(dct[name], start=1)) + "</li>"
            if name is None
            else "<li>" + ", ".join(parse_link((name, url)) for url in dct[name]) + "</li>"
        )
        for name in ordered_names
    ) + "</ul>"


def get_top_authors(top_k: int) -> tuple[str, int]:
    global TOP_K
    authors: list[tuple[str, str]] = []
    num_of_authors: list[int] = []
    for project in load_projects():
        normalized = normalize_authors(project)
        authors.extend(normalized)
        num_of_authors.append(len(normalized))
    cnt = Counter(authors)
    most_common = cnt.most_common()
    contributions = most_common[top_k][1]
    idx = top_k
    while idx < len(most_common) and most_common[idx][1] == contributions:
        idx += 1
    num_of_visible = int(min(mean(num_of_authors), median(num_of_authors))) if num_of_authors else 1
    TOP_K = idx
    return "<ul>" + " ".join(f"<li>[{author}]({link})</li>" for (author, link), _ in most_common[:idx]) + "</ul>", num_of_visible


def get_top_repos(top_k: int) -> str:
    repos: dict[str, int] = {}
    for project in load_projects():
        git_link = get_first_link(project, "git")
        stars = project.get("stars")
        if isinstance(git_link, str) and isinstance(stars, int):
            name, value = get_git_name_stars((git_link, stars))
            repos[name] = value
    repos_sorted = sorted(repos.items(), key=lambda f: f[1], reverse=True)[:top_k]
    return "<ul>" + " ".join(f"<li>{url.split('com/')[1].split('/')[1]}\t{git_url(url)}</li>" for url, _ in repos_sorted) + "</ul>"


def generate_cloud() -> None:
    def get_font_path(font_name: str = "Comic Sans MS"):
        for path in font_manager.findSystemFonts(fontext="ttf"):
            font = font_manager.FontProperties(fname=path)
            if font_name in font.get_name():
                return path
        return None

    with urlopen("https://img.icons8.com/color/1600/google-colab.png") as resp:
        image = asarray(bytearray(resp.read()), dtype="uint8")
    _, bw_img = threshold(imdecode(image, IMREAD_GRAYSCALE), 127, 255, THRESH_BINARY)
    mask = bw_img[any(bw_img, axis=1)]
    mask = mask[:, any(mask, axis=0)]
    text = " ".join(
        " ".join([str(p.get("name") or ""), str(p.get("description") or "")])
        for p in load_projects()
    )
    wc = WordCloud(mask=~mask, collocation_threshold=10, background_color="#0d1117", colormap="plasma", font_path=get_font_path())
    wc.generate(text)
    with open("images/cloud.svg", "w", encoding="utf-8") as f:
        f.write(wc.to_svg())


def get_top_papers(top_k: int) -> str:
    papers: dict[str, tuple[str, int]] = {}
    for project in load_projects():
        doi_link = get_first_link(project, "doi")
        citations = project.get("citations")
        if isinstance(doi_link, str) and isinstance(citations, int):
            if doi_link not in papers or citations > papers[doi_link][1]:
                papers[doi_link] = (project.get("name", ""), citations)
    repos = sorted([(name, url, citations) for url, (name, citations) in papers.items()], key=lambda f: f[2], reverse=True)[:top_k]
    return "<ul>" + " ".join(f"<li>{name}\t{doi_url(url)}</li>" for name, url, _ in repos) + "</ul>"


def get_best_of_the_best(authors: str, packages: list[tuple[str, int, int]], top_k: int) -> str:
    packages_str = "<ul>" + " ".join(f"<li>{package}\t{pypi_url(package)}</li>" for package, _, _ in sorted(packages, key=lambda p: p[2], reverse=True)[:top_k]) + "</ul>"
    return f"""| authors | repositories | papers | packages |
|---|---|---|---|
| {authors} | {get_top_repos(top_k)} | {get_top_papers(top_k)} | {packages_str} |"""


def generate_table(data: list[dict], num_visible_authors: int) -> list[str]:
    colabs = sorted(data, key=lambda kv: kv.get("update", 0), reverse=True)
    to_write = [
        "| name | description | authors | links | colaboratory | update |",
        "|------|-------------|:--------|:------|:------------:|:------:|",
    ]
    for line in colabs:
        row = dict(line)
        row["author"] = parse_authors(normalize_authors(row), num_visible_authors)
        row["links"] = parse_links(get_link_tuples(row))
        row["url"] = colab_url(row.get("colab", ""))
        update_ts = row.get("update")
        row["update"] = datetime.fromtimestamp(update_ts).strftime("%d.%m.%Y") if isinstance(update_ts, (int, float)) else ""
        to_write.append("| {name} | {description} | {author} | {links} | {url} | {update} |".format(**row))
    return to_write


def get_pypi_downloads(engine: str = "pypistats"):
    packages: set[str] = set()
    for project in load_projects():
        for key, url in get_link_tuples(project):
            if key == "pypi":
                package = url.rstrip("/").split("/")[-1]
                if package:
                    packages.add(package)
    if engine == "bigquery":
        def get_query(date_filtering: str) -> str:
            return f"""
            SELECT
                file.project,
                COUNT(*) AS num_downloads
            FROM
                `bigquery-public-data.pypi.file_downloads`
            WHERE
                file.project IN ('{"', '".join(packages)}')
                AND DATE(timestamp) BETWEEN {date_filtering}
            GROUP BY
                file.project
            """

        client = bigquery.Client()
        last_month = "DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 DAY)"
        total = "DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR) AND CURRENT_DATE()"
        query_last_month = client.query(get_query(last_month))
        query_total = client.query(get_query(total))
        # query_job = client.query(f"""
        #     SELECT
        #         file.project,
        #         COUNTIF(DATE(timestamp) BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH)
        #         AND DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 DAY)) AS num_downloads_last_month,
        #         COUNT(*) AS total_num_downloads
        #     FROM
        #         `bigquery-public-data.pypi.file_downloads`
        #     WHERE
        #         file.project IN ('{"', '".join(packages)}')
        #     GROUP BY
        #         file.project
        # """)
        res_last_month = {row.project: row.num_downloads for row in query_last_month.result()}
        res_total = {row.project: row.num_downloads for row in query_total.result()}

        return [(package, res_last_month[package], res_total[package]) for package in packages]
    res = []
    for package in tqdm(packages):
        try:
            recent_stats = recent(package, format="pandas")
            overall_stats = overall(package, format="pandas")
            last_month_downloads = recent_stats.last_month.item()
            total_downloads = overall_stats.query('category == "Total"').downloads.item()
            res.append((package, last_month_downloads, total_downloads))
        except Exception:
            res.append((package, 0, 0))
    return [(package, recent(package, format='pandas').last_month.item(), overall(package, format='pandas').query('category == "Total"').downloads.item()) for package in tqdm(packages)]


def get_trending(packages: list[tuple[str, int, int]], top_k: int):
    old_stars = read_json("data/stars.json")
    old_citations = read_json("data/citations.json")
    new_stars: dict[str, int] = {}
    new_citations: dict[str, tuple[str, int]] = {}

    for project in load_projects():
        git_link = get_first_link(project, "git")
        doi_link = get_first_link(project, "doi")
        stars = project.get("stars")
        citations = project.get("citations")
        if isinstance(git_link, str) and isinstance(stars, int):
            name, value = get_git_name_stars((git_link, stars))
            new_stars[name] = value
        if isinstance(doi_link, str) and isinstance(citations, int):
            new_citations[project.get("name", "")] = (doi_link, citations)

    trending_repos = sorted(new_stars, key=lambda url: new_stars[url] / old_stars.get(url, float("inf")), reverse=True)[:top_k]
    trending_papers = sorted(
        new_citations,
        key=lambda name: new_citations[name][1] / max(old_citations.get(name, ["", float("inf")])[1], 1),
        reverse=True,
    )[:top_k]
    trending_packages = sorted(packages, key=lambda p: p[1] / max((p[2] - p[1]), 1), reverse=True)[:top_k]

    repos_str = "<ul>" + " ".join(f"<li>{url.split('com/')[1].split('/')[1]}\t{git_url(url)}</li>" for url in trending_repos) + "</ul>"
    papers_str = "<ul>" + " ".join(f"<li>{name}\t{doi_url(new_citations[name][0])}</li>" for name in trending_papers) + "</ul>"
    packages_str = "<ul>" + " ".join(f"<li>{package}\t{pypi_url(package, period='dw')}</li>" for package, _, _ in trending_packages) + "</ul>"

    return f"""| repositories | papers | packages |
|---|---|---|
| {repos_str} | {papers_str} | {packages_str} |"""


def generate_markdown():
    projects = load_projects()
    top_authors, num_visible_authors = get_top_authors(TOP_K)
    packages = get_pypi_downloads()
    to_write = [
        "[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/amrzv/awesome-colab-notebooks)](https://hits.seeyoufarm.com)",
        "![awesome-colab-notebooks](https://count.getloli.com/get/@awesome-colab-notebooks?theme=rule34)\n",
        "[![Word Cloud](images/cloud.svg)](images/cloud.svg)",
        "\nThe page might not be rendered properly. Please open [README.md](https://github.com/amrzv/awesome-colab-notebooks/blob/main/README.md) file directly",
        "# Awesome colab notebooks collection for ML experiments",
        "## Trending",
        get_trending(packages, TOP_K),
        "## Projects",
        "<details>\n<summary>PROJECTS</summary>\n",
        *generate_table(projects, num_visible_authors),
        "\n</details>\n",
        "# Best of the best",
        get_best_of_the_best(top_authors, packages, TOP_K),
        "\n[![Stargazers over time](https://starchart.cc/amrzv/awesome-colab-notebooks.svg?variant=adaptive)](https://starchart.cc/amrzv/awesome-colab-notebooks)",
        "\n(generated by [generate_markdown.py](generate_markdown.py) based on [data.json](data/data.json))",
    ]
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(to_write))


def main():
    generate_cloud()
    generate_markdown()

if __name__ == '__main__':  
    main()
