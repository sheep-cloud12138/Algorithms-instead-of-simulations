import requests, time
H={"User-Agent":"Mozilla/5.0"}
# 测试 r.jina.ai 代理抓摘要
for name,u in [("jina-SD","https://r.jina.ai/https://www.sciencedirect.com/science/article/abs/pii/S0017931026X"),
               ("unpaywall","https://api.unpaywall.org/v2/10.1016/j.ijheatmasstransfer.2026.129230?email=researcher@example.com")]:
    try:
        r=requests.get(u,headers=H,timeout=45)
        print(f"{name}: {r.status_code} | {len(r.content)}B")
        print(r.text[:300].replace("\n"," "))
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}")
    print("---")
