import urllib.request, os, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
UA={"User-Agent":"Mozilla/5.0"}
base="https://huggingface.co/datasets/NurshatMenglik/NeuralConvection_2D_TrainTest/resolve/main/"
jobs=[("train_legacy/X.pt","data/train_legacy/X.pt"),("train_legacy/Y.pt","data/train_legacy/Y.pt"),
      ("train_legacy/per_sample.csv","data/train_legacy/per_sample.csv"),
      ("test_legacy/X.pt","data/test_legacy/X.pt"),("test_legacy/Y.pt","data/test_legacy/Y.pt"),
      ("test_legacy/per_sample.csv","data/test_legacy/per_sample.csv")]
for src,dst in jobs:
    if os.path.exists(dst) and os.path.getsize(dst)>0:
        print("skip",dst); continue
    t=time.time()
    req=urllib.request.Request(base+src,headers=UA)
    with urllib.request.urlopen(req,timeout=300) as r, open(dst,"wb") as f:
        while True:
            c=r.read(1<<22)
            if not c: break
            f.write(c)
    print(f"ok {dst} {os.path.getsize(dst)/1e6:.1f} MB in {time.time()-t:.0f}s")
