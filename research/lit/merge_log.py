import json, glob, os, datetime
log={"generated":datetime.datetime.now().isoformat(timespec="seconds"),
     "note":"期刊级文献排雷原始检索日志；每条均记录渠道/检索式/命中数/是否取得摘要"}
files=["scan_openalex.json","scan_s2.json","scan_s2b.json","scan_s2c.json","scan_s2d.json",
       "scan_epmc.json","scan_crossref.json","scan_arxiv.json","abstracts_key.json"]
out={}
for f in files:
    if os.path.exists(f):
        try: out[f]=json.load(open(f,encoding="utf-8"))
        except Exception as e: out[f]=f"READ_ERR {e}"
    else: out[f]=None
json.dump(out,open("raw_log_all.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)
print("wrote raw_log_all.json", os.path.getsize("raw_log_all.json"),"B")
for f in files: print(" ",f,"OK" if os.path.exists(f) else "MISSING", os.path.getsize(f) if os.path.exists(f) else "")
