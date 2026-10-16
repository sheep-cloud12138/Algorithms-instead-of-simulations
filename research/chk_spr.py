import requests, time
H={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36"}
r=requests.get("https://link.springer.com/search?query=radiation+natural+convection+surrogate",headers=H,timeout=20)
print("springer len",len(r.text)); print(r.text[:600].replace("\n"," "))
print("----CORE----")
r2=requests.get("https://api.core.ac.uk/v3/search/works/",headers=H,timeout=25,params={"q":"radiation natural convection surrogate","limit":3})
print(r2.status_code, r2.text[:400])
