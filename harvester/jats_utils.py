from lxml import etree
METHOD={'experimental','methods','materials and methods','synthesis','preparation','procedure'}

def jats_to_sections(x):
    root=etree.fromstring(x.encode('utf-8'))
    secs=[]
    for sec in root.xpath('.//sec'):
        head_el=sec.find('title'); head=head_el.text if head_el is not None else ''
        text=' '.join(sec.xpath('.//text()')).strip()
        if text: secs.append({'heading':head,'text':text})
    return secs

def filter_methods_sections(sections):
    out=[s for s in sections if any(k in (s.get('heading') or '').lower() for k in METHOD)]
    return out or sections[:2]
