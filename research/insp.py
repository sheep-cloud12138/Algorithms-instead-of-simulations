import torch, io, sys, numpy as np, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
X=torch.load("data/train_legacy/X.pt", map_location="cpu", mmap=True)
Y=torch.load("data/train_legacy/Y.pt", map_location="cpu", mmap=True)
print("X", type(X).__name__, getattr(X,"shape",None), getattr(X,"dtype",None))
print("Y", type(Y).__name__, getattr(Y,"shape",None), getattr(Y,"dtype",None))
with open("data/train_legacy/per_sample.csv", encoding="utf-8") as f:
    r=csv.reader(f); hdr=next(r); rows=[next(r) for _ in range(3)]
print("CSV HEADERS:", len(hdr))
for i,h in enumerate(hdr): print(f"   [{i}] {h}")
print("ROW0:", rows[0][:24])
print("ROW1:", rows[1][:24])
