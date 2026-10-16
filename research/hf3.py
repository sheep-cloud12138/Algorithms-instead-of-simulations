import urllib.request, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0"}
base="https://huggingface.co/datasets/NurshatMenglik/NeuralConvection_2D_TrainTest/resolve/main/"
files=["train_legacy/X.pt","train_legacy/Y.pt","train_legacy/per_sample.csv","test_legacy/X.pt","test_legacy/Y.pt","test_legacy/per_sample.csv","train/X.pt","test_ood/X.pt"]
tot=0
for f in files:
    req=urllib.request.Request(base+f,headers=UA,method="HEAD")
    try:
        r=urllib.request.urlopen(req,timeout=60)
        n=int(r.headers.get("Content-Length",0)); tot+=n
        print(f"{n/1e6:10.1f} MB  {f}")
    except Exception as e:
        print(f"      ERR  {f}  {type(e).__name__}")
print(f"{tot/1e6:10.1f} MB  TOTAL")
