import urllib.request, re
for aid in ["2410.05894","2511.15856","2601.20265"]:
    req=urllib.request.Request("https://arxiv.org/abs/"+aid, headers={"User-Agent":"Mozilla/5.0"})
    h=urllib.request.urlopen(req,timeout=40).read().decode("utf-8","ignore")
    c=re.search(r'tablecell comments">(.*?)</td>', h, re.S)
    j=re.search(r'tablecell jref">(.*?)</td>', h, re.S)
    def clean(x): return " ".join(re.sub(r"<[^>]+>"," ",x).split()) if x else None
    print(aid, "| COMMENTS:", clean(c.group(1) if c else None), "| JREF:", clean(j.group(1) if j else None))
