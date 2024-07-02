from collections import Counter, defaultdict
from datetime import datetime
from json import load
from numpy import mean, median
from os.path import join

badges = {'colab', 'yt', 'git', 'wiki', 'kaggle', 'arxiv', 'tf', 'pt', 'medium', 'reddit', 'neurips', 'pwc', 'hf', 'docs', 'slack', 'twitter', 'deepmind', 'discord', 'docker'}

TOP_K = 20

def colab_url(url: str) -> str:
    return f'[![Open In Colab](images/colab.svg)]({url})'

def doi_url(url: str) -> str:
    doi = url.split('org/')[1]
    return f'[![](https://api.juleskreuer.eu/citation-badge.php?doi={doi})]({url})'

def git_url(url: str) -> str:
    repo = '/'.join(url.split('com/')[1].split('/')[:2])
    return f'[![](https://img.shields.io/github/stars/{repo}?style=social)]({url})'

def read_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return load(f)

def parse_link(link_tuple: list[list[str]], height=20) -> str:
    name, url = link_tuple
    if name in badges:
        return f'[<img src="images/{name}.svg" alt="{name}" height={height}/>]({url})'
    return f'[{name}]({url})'

def parse_authors(authors: list[tuple[str, str]], num_of_visible: int) -> str:
    num_authros = len(authors)
    if len(authors) == 1:
        return '[{}]({})'.format(*authors[0])
    if len(authors) <= num_of_visible + 1:
        return '<ul>' + ' '.join(f'<li>[{author}]({link})</li>' for author,link in authors[:num_of_visible + 1]) + '</ul>'
    return '<ul>' + ' '.join(f'<li>[{author}]({link})</li>' for author,link in authors[:num_of_visible]) + '<details><summary>others</summary>' + ' '.join(f'<li>[{author}]({link})</li>' for author,link in authors[num_of_visible:]) + '</ul></details>'

def parse_links(list_of_links: list[tuple[str, str]]) -> str:
    if len(list_of_links) == 0:
        return ''
    dct = defaultdict(list)
    for tupl in list_of_links:
        name, url = tupl[0], tupl[1]
        dct[name].append(url)
    line = ''
    if 'doi' in dct:
        line += doi_url(dct['doi'][0]) + ' '
        dct.pop('doi')
    if 'git' in dct:
        line += git_url(dct['git'][0]) + ' '
        if len(dct['git']) == 1:
            dct.pop('git')
        else:
            dct['git'].pop(0)
    if len(dct) == 0:
        return line

    return line + '<ul>' + ''.join('<li>' + ', '.join(parse_link((name, url)) for url in dct[name]) + '</li>' for name in dct.keys()) + '</ul>'

def get_top_authors(topK) -> tuple[str, int]:
    global TOP_K
    research = read_json(join('data', 'research.json'))
    tutorials = read_json(join('data', 'tutorials.json'))
    
    authors, num_of_authors = [], []
    for project in research + tutorials:
        authors.extend([tuple(author) for author in project['author']])
        num_of_authors.append(len(project['author']))
    cnt = Counter(authors)
    most_common = cnt.most_common()
    contributions = most_common[topK][1]
    idx = topK
    while idx < len(most_common) and most_common[idx][1] == contributions:
        idx += 1
    num_of_visible = int(min(mean(num_of_authors), median(num_of_authors)))
    TOP_K = idx
    
    return '<ul>' + ' '.join(f'<li>[{author}]({link})</li>' for (author,link),_ in most_common[:idx]) + '</ul>', num_of_visible

def get_top_repos(topK) -> str:
    research = read_json(join('data', 'research.json'))
    tutorials = read_json(join('data', 'tutorials.json'))
    repos = {}
    for project in research + tutorials:
        for link in project['links']:
            if link[0] == 'git':
                _, url, stars = link
                idx = url.index('/', 19) + 1
                idx = url.find('/', idx)
                key = url[:idx] if idx != -1 else url
                repos[key] = stars
                break
    repos = sorted(repos.items(), key=lambda f: f[1], reverse=True)[:topK]
    
    return '<ul>' + ' '.join(f"<li>{'/'.join(url.split('com/')[1].split('/')[:2])}\t{git_url(url)}</li>" for url,_ in repos) + '</ul>'

def get_top_papers(topK) -> str:
    research = read_json(join('data', 'research.json'))
    tutorials = read_json(join('data', 'tutorials.json'))
    repos = {}
    for project in research + tutorials:
        for link in project['links']:
            if link[0] == 'doi':
                if link[1] not in repos or link[2] > repos[link[1]][1]:
                    repos[link[1]] = (project['name'], link[2])
                break
    repos = sorted([(name, url, citations) for url, (name, citations) in repos.items()], key=lambda f: f[2], reverse=True)[:topK]
    
    return '<ul>' + ' '.join(f"<li>{name}\t{doi_url(url)}</li>" for name,url,_ in repos) + '</ul>'

def get_best_of_the_best(authors: str, topK: int) -> str:
    table = f'''| authors | repositories | papers |
|---|---|---|
| {authors} | {get_top_repos(topK)} | {get_top_papers(topK)} |'''
    return table

def generate_table(fn: str, num_visible_authors: int):
    data = read_json(fn)
    colabs = sorted(data, key=lambda kv: kv['update'], reverse=True)
    to_write = [
        '| name | description | authors | links | colaboratory | update |',
        '|------|-------------|:--------|:------|:------------:|:------:|',
    ]
    for line in colabs:
        line['author'] = parse_authors(line['author'], num_visible_authors)
        line['links'] = parse_links(sorted(line['links'], key=lambda x: x[0]))
        line['url'] = colab_url(line['colab'])
        line['update'] = datetime.fromtimestamp(line['update']).strftime('%d.%m.%Y')
        to_write.append('| {name} | {description} | {author} | {links} | {url} | {update} |'.format(**line))
    return to_write

def get_trending(topK: int):
    old_stars = read_json('data/stars.json')
    old_citations = read_json('data/citations.json')
    research = read_json(join('data', 'research.json'))
    tutorials = read_json(join('data', 'tutorials.json'))
    new_stars, new_citations = {}, {}
    for project in research + tutorials:
        used = set()
        for link in project['links']:
            if link[0] == 'git' and 'git' not in used:
                _, url, stars = link
                idx = url.index('/', 19) + 1
                idx = url.find('/', idx)
                key = url[:idx] if idx != -1 else url
                new_stars[key] = stars
                used.add('git')
            elif link[0] == 'doi' and 'doi' not in used:
                _, url, citations = link
                new_citations[project['name']] = (url, citations)
                used.add('doi')
    trending_repos = sorted(new_stars, key=lambda url: new_stars[url] / old_stars.get(url, float('inf')), reverse=True)[:topK]
    trending_papers = sorted(new_citations, key=lambda name: new_citations[name][1] / max(old_citations.get(name, ['', float('inf')])[1], 1), reverse=True)[:topK]
    repos_str = '<ul>' + ' '.join(f"<li>{'/'.join(url.split('com/')[1].split('/')[:2])}\t{git_url(url)}</li>" for url in trending_repos) + '</ul>'
    papers_str = '<ul>' + ' '.join(f"<li>{name}\t{doi_url(new_citations[name][0])}</li>" for name in trending_papers) + '</ul>'
    
    return f'''| repositories | papers |
|---|---|
| {repos_str} | {papers_str} |'''

def generate_markdown():
    top_authors, num_visible_authors = get_top_authors(TOP_K)
    to_write = [
        '[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/amrzv/awesome-colab-notebooks)](https://hits.seeyoufarm.com)',
        '![awesome-colab-notebooks](https://count.getloli.com/get/@awesome-colab-notebooks?theme=rule34)',
        '\nThe page might not be rendered properly. Please open [README.md](https://github.com/amrzv/awesome-colab-notebooks/blob/main/README.md) file directly',
        '# Awesome colab notebooks collection for ML experiments',
        '## Trending',
        get_trending(TOP_K),
        '## Research',
        *generate_table(join('data', 'research.json'), num_visible_authors),
        '## Tutorials',
        *generate_table(join('data', 'tutorials.json'), num_visible_authors),
        '# Best of the best',
        get_best_of_the_best(top_authors, TOP_K),
        '\n[![Stargazers over time](https://starchart.cc/amrzv/awesome-colab-notebooks.svg)](https://starchart.cc/amrzv/awesome-colab-notebooks)',
        '\n(generated by [generate_markdown.py](generate_markdown.py) based on [research.json](data/research.json) and [tutorials.json](data/tutorials.json))'
    ]
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(to_write))

def main():
    generate_markdown()

if __name__ == '__main__':  
    main()
