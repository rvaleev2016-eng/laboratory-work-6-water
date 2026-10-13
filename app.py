from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
app=FastAPI(title="WATER ITSM + Retrieval Demonstrator",version="1.0",description="Локальный проверяемый демонстратор. TF-IDF retrieval; не является полноценной LLM RAG.")
ROOT=Path(__file__).resolve().parent
KB=json.loads((ROOT/"kb.json").read_text(encoding="utf-8"))
TICKETS=json.loads((ROOT/"tickets.json").read_text(encoding="utf-8"))
vec=TfidfVectorizer(analyzer="char",ngram_range=(2,4))
matrix=vec.fit_transform([r["title"]+" "+r["content"] for r in KB])
class Ask(BaseModel):
    query:str
    k:int=3
class NewTicket(BaseModel):
    title:str
    priority:str="medium"
@app.get("/health")
def health():return {"status":"ok","articles":len(KB),"tickets":len(TICKETS)}
@app.post("/search")
def search(req:Ask):
    if not req.query.strip():raise HTTPException(422,"empty query")
    scores=cosine_similarity(vec.transform([req.query]),matrix)[0]
    order=sorted(range(len(KB)),key=lambda i:float(scores[i]),reverse=True)[:max(1,min(req.k,10))]
    return {"sources":[{**KB[i],"score":round(float(scores[i]),4)} for i in order]}
@app.post("/ask")
def ask(req:Ask):
    found=search(req)["sources"]
    return {"answer":found[0]["content"] if found and found[0]["score"]>0 else "Нет уверенного совпадения: создайте заявку.","sources":found,"generator":"extractive-demo"}
@app.get("/tickets")
def tickets():return TICKETS
@app.post("/tickets")
def create(req:NewTicket):
    if req.priority not in ("critical","high","medium","low"):raise HTTPException(422,"invalid priority")
    item={"id":f"DEMO-{len(TICKETS)+1:03d}","title":req.title,"priority":req.priority,"status":"New"}
    TICKETS.append(item);return item
@app.get("/metrics")
def metrics():
    total=len(TICKETS); breached=sum(x.get("sla_breached")=="Да" for x in TICKETS)
    return {"total":total,"sla_breach_rate_percent":round(100*breached/total,1) if total else 0,"source":"provided demonstration CSV"}
@app.post("/pipeline")
def pipeline(req:Ask):
    r=ask(req);conf=r["sources"][0]["score"] if r["sources"] else 0
    if conf<0.2:
        ticket=create(NewTicket(title=req.query,priority="medium"))
        return {"route":"itsm","ticket":ticket,"retrieval":r}
    return {"route":"knowledge_base","retrieval":r}


# --- MOLSP / PuLP-CBC ---
from scheduler import optimize

class ScheduleRequest(BaseModel):
    slots:int=8
    balance_lambda:float=10.0

@app.post("/molsp",tags=["MOLSP optimizer"])
def molsp(req:ScheduleRequest):
    try:
        return optimize([t for t in TICKETS if t.get("status","").lower() not in ("resolved","closed")],slots=req.slots,balance_lambda=req.balance_lambda)
    except ValueError as exc:
        raise HTTPException(422,str(exc))

@app.post("/pipeline/molsp",tags=["MOLSP optimizer"])
def pipeline_molsp(req:Ask):
    result=pipeline(req)
    result["molsp"]=optimize([t for t in TICKETS if t.get("status","").lower() not in ("resolved","closed")])
    return result


@app.post("/molsp/demo20",tags=["MOLSP optimizer"])
def molsp_demo20(req:ScheduleRequest):
    """Demonstration with all 20 synthetic tickets, including previously resolved."""
    return optimize(TICKETS[:20],slots=req.slots,balance_lambda=req.balance_lambda)
