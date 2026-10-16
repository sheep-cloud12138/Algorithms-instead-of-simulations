import urllib.request, json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0"}
def get(url):
    req=urllib.request.Request(url,headers=UA)
    try:
        r=urllib.request.urlopen(req,timeout=60); return json.loads(r.read())
    except Exception as e:
        return {"__err__": f"{type(e).__name__}: {e}"}
for ds in ["NurshatMenglik/NeuralConvection_2D_TrainTest","NurshatMenglik/NeuralConvection_3D_TrainTest"]:
    d=get("https://huggingface.co/api/datasets/"+ds)
    print("="*95); print(ds)
    if "__err__" in d: print("  ", d["__err__"]); continue
    print("  lastModified:", d.get("lastModified"), "| downloads:", d.get("downloads"))
    sib=d.get("siblings") or []
    print("  files:", len(sib))
    for s in sib[:40]:
        print("   -", s.get("rfilename"))
