#!/usr/bin/env python3
"""Fetch authoritative metadata (full author lists, titles, dates) from the arXiv API for every arXiv id in the catalog."""
import json, re, time, urllib.request, xml.etree.ElementTree as ET, os
ROOT = os.path.dirname(os.path.abspath(__file__))
cat = json.load(open(os.path.join(ROOT, 'catalog.json')))
ids = sorted({(x.get('arxiv') or '').strip() for x in cat['methods'] + cat['papers'] if re.match(r'^\d{4}\.\d{4,5}$|^[a-z\-]+/\d{7}$', (x.get('arxiv') or '').strip())})
out_path = os.path.join(ROOT, 'arxiv', 'meta.json')
meta = json.load(open(out_path)) if os.path.exists(out_path) else {}
todo = [i for i in ids if i not in meta]
print(len(ids), 'ids;', len(todo), 'to fetch')
ns = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
for k in range(0, len(todo), 80):
    chunk = todo[k:k+80]
    url = 'https://export.arxiv.org/api/query?id_list=' + ','.join(chunk) + '&max_results=100'
    xml = None
    for attempt in range(6):
        try:
            xml = urllib.request.urlopen(url, timeout=60).read(); break
        except Exception as e:
            print('retry', e); time.sleep(10 * (attempt + 1))
    if xml is None:
        print('giving up on this chunk for now:', chunk[:3], '...'); continue
    root = ET.fromstring(xml)
    for e in root.findall('a:entry', ns):
        aid = e.find('a:id', ns).text.rsplit('/abs/', 1)[-1]
        aid = re.sub(r'v\d+$', '', aid)
        meta[aid] = {
            'title': re.sub(r'\s+', ' ', e.find('a:title', ns).text).strip(),
            'authors': [a.find('a:name', ns).text for a in e.findall('a:author', ns)],
            'published': e.find('a:published', ns).text[:10],
            'journal_ref': (e.find('arxiv:journal_ref', ns).text if e.find('arxiv:journal_ref', ns) is not None else ''),
            'comment': (e.find('arxiv:comment', ns).text if e.find('arxiv:comment', ns) is not None else ''),
            'primary': e.find('arxiv:primary_category', ns).get('term'),
        }
    json.dump(meta, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print('fetched', min(k+80, len(todo)), '/', len(todo)); time.sleep(3.5)
missing = [i for i in ids if i not in meta]
print('missing', missing)
