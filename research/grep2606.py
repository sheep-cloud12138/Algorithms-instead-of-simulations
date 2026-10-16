import re
t = open("research/paper_2606.txt", encoding="utf-8", errors="ignore").read()
t = " ".join(t.split())
pats = [r".{300}out-of-distribution.{400}", r".{200}1e2.{200}", r".{200}10\^\{?2\}?.{300}", r".{250}extrapolat.{350}", r".{200}three-dimensional results.{400}", r".{200}two orders of magnitude.{200}", r".{250}orders of magnitude.{250}"]
seen=set()
for p in pats:
    for m in re.finditer(p, t, re.I):
        s = m.group(0)
        if s[:80] in seen: continue
        seen.add(s[:80])
        print("---"); print(s)
