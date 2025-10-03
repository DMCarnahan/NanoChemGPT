from lxml import etree

METHOD = {
    "experimental",
    "methods",
    "materials and methods",
    "synthesis",
    "preparation",
    "procedure",
}


def tei_to_sections(x):
    root = etree.fromstring(x.encode("utf-8"))
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    secs = []
    for div in root.xpath('.//t:div[@type="section"]', namespaces=ns):
        head = (div.xpath("./t:head/text()", namespaces=ns) or [""])[0].strip()
        text = " ".join(div.xpath(".//text()", namespaces=ns)).strip()
        if text:
            secs.append({"heading": head, "text": text})
    return secs


def filter_methods_sections(sections):
    out = [
        s
        for s in sections
        if any(k in (s.get("heading") or "").lower() for k in METHOD)
    ]
    return out or sections[:2]
