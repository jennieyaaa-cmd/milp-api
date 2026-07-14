from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict

app = FastAPI()

# Definisikan input JSON dari Google Sheets
class OptimizationInput(BaseModel):
    t_pabrik: float
    t_retailer: float
    T_max: float
    mobil_capacities: List[int]  # [20, 25]
    demand: Dict[str, int]       # {"1": 3, "2": 2, ...} "retailer_id": demand_keranjang
    t_matrix: List[List[float]]  # Matriks waktu fix 21x21 (indeks 0 adalah Pabrik)

@app.post("/solve-route")
def solve_route(data: OptimizationInput):
    try:
        # 1. Setup Gurobi Environment menggunakan WLS License dari Colab kamu
        # (Isi dengan kredensial WLS Gurobi kamu yang ada di file MILP)
        params = {
            "WLSAccessID": "AKU_AKAN_BANTU_AMANKAN_INI", 
            "WLSSecret": "SECRET_KAMU",
            "LICENSEID": 2818118,
        }
        env = gp.Env(params=params)
        model = gp.Model("MTVRP_Direct", env=env)
        model.setParam('TimeLimit', 30)
        model.setParam('OutputFlag', 0)

        # 2. Setup Parameter & Set
        V = list(range(1, 21))  # 20 Retailer
        K = list(range(1, len(data.mobil_capacities) + 1)) # [1, 2]
        Q = {k: data.mobil_capacities[k-1] for k in K}
        
        t_pabrik = data.t_pabrik
        t_retailer = data.t_retailer
        T_max = data.T_max
        M = 10000

        # Konversi demand key dari string ke integer
        current_demand = {i: data.demand.get(str(i), 0) for i in V}

        # Matriks waktu t_input
        t_input = {}
        for i in range(21):
            for j in range(21):
                t_input[i, j] = data.t_matrix[i][j]

        # 3. Variabel Keputusan
        x = model.addVars([(i, j, k) for i in V for j in V if i != j for k in K], vtype=GRB.BINARY, name="x")
        x_refill = model.addVars([(i, j, k) for i in V for j in V if i != j for k in K], vtype=GRB.BINARY, name="x_refill")
        start = model.addVars(V, K, vtype=GRB.BINARY, name="start")
        end = model.addVars(V, K, vtype=GRB.BINARY, name="end")
        W = model.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, name="W")
        Y = model.addVars(V, K, vtype=GRB.CONTINUOUS, lb=0, name="Y")

        # Batasan Kapasitas (Upper Bound)
        for i in V:
            for k in K:
                Y[i, k].ub = Q[k]

        # 4. Objective Function (Waktu Operasional Minimal)
        model.setObjective(
            gp.quicksum((t_input[i, j] + t_retailer) * x[i, j, k] for i in V for j in V if i != j for k in K) +
            gp.quicksum((t_input[i, 0] + t_pabrik + t_input[0, j] + t_retailer) * x_refill[i, j, k] for i in V for j in V if i != j for k in K) +
            gp.quicksum((t_pabrik + t_input[0, i]) * start[i, k] for i in V for k in K) +
            gp.quicksum(t_input[i, 0] * end[i, k] for i in V for k in K),
            GRB.MINIMIZE
        )

        # 5. Batasan Kendala (Constraints) dari MILP.pdf
        # --- Kendala Aliran Rute ---
        model.addConstrs(
            gp.quicksum(x[i, j, k] + x_refill[i, j, k] for i in V if i != j for k in K) +
            gp.quicksum(start[j, k] for k in K) == 1 for j in V
        )
        model.addConstrs(
            gp.quicksum(x[i, j, k] + x_refill[i, j, k] for i in V if i != j) + start[j, k] ==
            gp.quicksum(x[j, i, k] + x_refill[j, i, k] for i in V if i != j) + end[j, k] for j in V for k in K
        )
        model.addConstrs(gp.quicksum(start[i, k] for i in V) <= 1 for k in K)
        model.addConstrs(gp.quicksum(end[i, k] for i in V) <= 1 for k in K)

        # --- Kendala Kapasitas & Subtour Elimination (MTZ Modifikasi) ---
        model.addConstrs(Y[i, k] <= Q[k] - current_demand[i] + M * (1 - start[i, k]) for i in V for k in K)
        model.addConstrs(Y[j, k] <= Y[i, k] - current_demand[j] + M * (1 - x[i, j, k]) for i in V for j in V if i != j for k in K)

        # --- Kendala Waktu Operasional ---
        model.addConstrs(W[i, k] >= t_pabrik + t_input[0, i] - M * (1 - start[i, k]) for i in V for k in K)
        model.addConstrs(W[j, k] >= W[i, k] + t_retailer + t_input[i, j] - M * (1 - x[i, j, k]) for i in V for j in V if i != j for k in K)
        model.addConstrs(W[j, k] >= (t_input[i, 0] + t_pabrik + t_input[0, j]) - M * (1 - x_refill[i, j, k]) for i in V for j in V if i != j for k in K)
        model.addConstrs(W[i, k] + t_retailer + t_input[i, 0] <= T_max + M * (1 - end[i, k]) for i in V for k in K)

        # 6. Optimasi
        model.optimize()

        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            total_waktu = round(model.ObjVal, 2)
            routes = {}

            # Rekonstruksi Rute untuk Tiap Mobil
            for k in K:
                start_node = next((i for i in V if start[i, k].x > 0.5), None)
                if start_node is not None:
                    route_list = [f"R-{start_node}"]
                    curr = start_node
                    while True:
                        nxt_direct = next((j for j in V if curr != j and (curr, j, k) in x and x[curr, j, k].x > 0.5), None)
                        nxt_refill = next((j for j in V if curr != j and (curr, j, k) in x_refill and x_refill[curr, j, k].x > 0.5), None)
                        
                        if nxt_direct is not None:
                            route_list.append(f"R-{nxt_direct}")
                            curr = nxt_direct
                        elif nxt_refill is not None:
                            route_list.append("[REFILL]")
                            route_list.append(f"R-{nxt_refill}")
                            curr = nxt_refill
                        else:
                            break
                    route_list.append("Selesai")
                    routes[f"mobil_{k}"] = " -> ".join(route_list)
                else:
                    routes[f"mobil_{k}"] = "Tidak digunakan"

            return {
                "status": "success",
                "total_waktu": total_waktu,
                "rute": routes
            }
        else:
            return {"status": "infeasible", "message": "Solusi tidak ditemukan."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))