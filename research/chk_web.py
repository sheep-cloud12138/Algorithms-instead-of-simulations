import requests,time
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36"}
for name,url in [
 ("SD-IJHMT","https://www.sciencedirect.com/journal/international-journal-of-heat-and-mass-transfer"),
 ("SD-search","https://www.sciencedirect.com/search?qs=surface%20radiation%20natural%20convection%20surrogate"),
 ("SpringerLink","https://link.springer.com/search?query=radiation+natural+convection+surrogate"),
 ("CORE","https://api.core.ac.uk/v3/search/works?q=test"),
 ("DOAJ","https://doaj.org/api/search/articles/radiation%20natural%20convection%20surrogate"),
 ("IJHMT-alt","https://api.elsevier.com/content/journal/ijhmt"),
]:
    try:
        r=requests.get(url,headers=H,timeout=20,allow_redirects=True)
        print(f"{name}: {r.status_code} | {len(r.content)}B | {r.url[:90]}")
    except Exception as e:
        print(f"{name}: FAIL {type(e).__name__}")
    time.sleep(1)
