"""MILP/MOLSP multiobjective scheduling using PuLP/CBC."""
from dataclasses import dataclass
import pulp

WEIGHTS={"critical":100,"high":50,"medium":20,"low":5}
LEVELS={"critical":{"L3"},"high":{"L2","L3"},"medium":{"L1","L2","L3"},"low":{"L1","L2","L3"}}

@dataclass
class Worker:
    id:str
    name:str
    level:str
    max_slots:int=8


def optimize(tickets, workers=None, slots=8, balance_lambda=10.0):
    """One-hour tasks with SLA deadlines, priority penalty, load balance.
    Objective minimizes weighted lateness + weighted nonassignment + max worker load.
    """
    if workers is None:
        workers=[Worker("W1","Специалист 1","L1"),Worker("W2","Специалист 2","L2"),Worker("W3","Специалист 3","L3")]
    if not 1<=slots<=24:raise ValueError("slots must be between 1 and 24")
    items=[]
    for k,t in enumerate(tickets):
        p=str(t.get("priority","medium")).lower()
        if p not in WEIGHTS:raise ValueError("Unknown priority: "+p)
        # End of permitted hour for response (0-indexed)
        deadline={"critical":1,"high":3,"medium":5,"low":slots-1}[p]
        items.append((k,str(t.get("id",k)),str(t.get("title","")),p,deadline))
    model=pulp.LpProblem("MOLSP",pulp.LpMinimize)
    x={}
    for k,tid,title,p,deadline in items:
        for w in workers:
            if w.level not in LEVELS[p]:continue
            for s in range(slots):x[k,w.id,s]=pulp.LpVariable(f"x_{k}_{w.id}_{s}",cat="Binary")
    for k,*_ in items:
        model+=pulp.lpSum(v for (i,_,_),v in x.items() if i==k)<=1
    for w in workers:
        for s in range(slots):model+=pulp.lpSum(v for (i,j,t),v in x.items() if j==w.id and t==s)<=1
        model+=pulp.lpSum(v for (i,j,t),v in x.items() if j==w.id)<=w.max_slots
    max_load=pulp.LpVariable("max_load",lowBound=0,cat="Integer")
    for w in workers:model+=max_load>=pulp.lpSum(v for (i,j,t),v in x.items() if j==w.id)
    # Nonassignment penalty makes high-priority work preferable under capacity constraints.
    missed=pulp.lpSum(WEIGHTS[p]*10*(1-pulp.lpSum(v for (i,j,s),v in x.items() if i==k)) for k,tid,title,p,deadline in items)
    late=pulp.lpSum(WEIGHTS[p]*max(0,s-deadline)*v for (k,j,s),v in x.items() for i,tid,title,p,deadline in items if i==k)
    model+=missed+late+balance_lambda*max_load
    solver=pulp.PULP_CBC_CMD(msg=False,timeLimit=30)
    model.solve(solver)
    status=pulp.LpStatus[model.status]
    assigned=[];done=set();loads={w.id:0 for w in workers}
    for (k,wid,s),var in x.items():
        if (pulp.value(var) or 0)>0.5:
            _,tid,title,p,deadline=items[k]
            assigned.append({"ticket_id":tid,"title":title,"priority":p,"worker_id":wid,"worker_name":next(w.name for w in workers if w.id==wid),"slot":s,"sla_ok":s<=deadline})
            done.add(k);loads[wid]+=1
    assigned.sort(key=lambda a:(a["slot"],a["worker_id"]))
    return {"status":status.lower(),"solver":"PuLP/CBC","assignments":assigned,"unassigned":[tid for k,tid,_,_,_ in items if k not in done],"loads":loads,"max_load":max(loads.values(),default=0),"objective":round(float(pulp.value(model.objective) or 0),2),"slots":slots,"balance_lambda":balance_lambda}
