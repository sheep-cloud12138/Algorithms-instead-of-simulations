import urllib.request, io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0"}
base="https://huggingface.co/datasets/NurshatMenglik/NeuralConvection_2D_TrainTest/resolve/main/"
def get(u, n=9000):
    req=urllib.request.Request(u,headers=UA)
    try:
        r=urllib.request.urlopen(req,timeout=90)
        return r.read().decode("utf-8","replace")[:n]
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"
print("### README.md")
print(get(base+"README.md"))
print()
for f in ["train_legacy/MANIFEST.json","test_legacy/MANIFEST.json","train/MANIFEST.json"]:
    print("### "+f)
    print(get(base+f, 2500))
    print()
