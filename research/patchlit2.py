import os, re
p = os.path.join("research","lit_review.py")
s = open(p, encoding="utf-8").read()
s = s.replace('''    p = os.path.join(OUT, f"batch{batch}.txt")
    open(p, "w", encoding="utf-8").write("\\n".join(lines))''',
'''    p = os.path.join(OUT, f"batch{batch}.txt")
    open(p, "w", encoding="utf-8").write("\\n".join(lines))''')
# make per-query robust + incremental flush
s = s.replace('''        res = q(query, n)
        if res is None:''', '''        try:
            res = q(query, n)
        except Exception as e:
            res = None; lines.append("   (EXC %s)" % e)
        p = os.path.join(OUT, f"batch{batch}.txt")
        open(p, "w", encoding="utf-8").write("\\n".join(lines))
        if res is None:''')
open(p,"w",encoding="utf-8").write(s)
print("ok")
