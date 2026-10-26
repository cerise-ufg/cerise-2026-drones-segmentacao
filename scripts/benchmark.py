"""Benchmark dos metodos de particao: 4 cenarios x N=2..6 x frota homogenea/heterogenea.

Uso: python scripts/benchmark.py [--ablacao]
Saidas: resultados/benchmark.csv, resultados/benchmark.json (e ablacao.csv com --ablacao)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from cerise_drones.cenarios import CENARIOS, frota_padrao  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402

RES = RAIZ / "resultados"
RES.mkdir(exist_ok=True)

METODOS = [
    ("faixas", "faixas", {}),
    ("kmeans", "kmeans", {}),
    ("darp", "darp", {}),
    ("darp+ref", "darp", {"refinar": 4, "sementes": "centros"}),
]
ABLACAO = [
    ("darp mult. (original)", "darp", {"pesagem": "multiplicativa"}),
    ("darp aditiva, bases", "darp", {}),
    ("darp aditiva, centros", "darp", {"sementes": "centros"}),
    ("darp pesos=capacidade", "darp", {"sementes": "centros", "modo_pesos": "capacidade"}),
    ("darp pesos=iguais", "darp", {"sementes": "centros", "modo_pesos": "igual"}),
    ("darp+ref (centros)", "darp", {"sementes": "centros", "refinar": 4}),
    ("faixas+ref", "faixas", {"refinar": 4}),
]
CAMPOS = ["cenario", "n", "frota", "metodo", "makespan_s", "tempo_medio_s", "cv_tempo", "cv_area",
          "erro_area_vs_alvo_max_pct", "distancia_total_m", "curvas_total", "cobertura_pct",
          "sobreposicao_entre_drones_pct", "n_voos_total", "todos_viaveis_um_voo", "regioes_desconexas",
          "tempo_particao_s", "tempo_cobertura_s", "tempo_algoritmo_s", "area_livre_m2"]


def rodar(lista, cenarios, ns, frotas, arq_csv, arq_json):
    linhas, completo = [], []
    # aquecimento (importacoes/threads do scikit-learn nao entram no tempo medido)
    a0 = CENARIOS["convexo"]()
    planejar_missao(a0, frota_padrao(a0, 2, False), "kmeans")
    t_ini = time.time()
    for cen in cenarios:
        area = CENARIOS[cen]()
        for n in ns:
            for het in frotas:
                frota = frota_padrao(area, n, het)
                for rot, met, kw in lista:
                    p = planejar_missao(area, frota, met, **kw)
                    m = p.metricas
                    lin = {k: m[k] for k in CAMPOS if k in m}
                    lin.update({"cenario": cen, "n": n, "frota": "heterogenea" if het else "homogenea", "metodo": rot})
                    linhas.append(lin)
                    completo.append({**lin, "por_drone": m["por_drone"], "historico_makespan_s": m.get("historico_makespan_s"),
                                     "darp": m.get("darp")})
                    print(f"[{time.time() - t_ini:6.0f}s] {cen:10s} N={n} {lin['frota'][:4]} {rot:22s} mk={m['makespan_s'] / 60:6.2f}min "
                          f"cv={m['cv_tempo']:.3f} dist={m['distancia_total_m'] / 1000:6.2f}km curvas={m['curvas_total']:3d} "
                          f"cob={m['cobertura_pct']:6.2f}% voos={m['n_voos_total']} t={m['tempo_algoritmo_s']:.2f}s", flush=True)
                with open(arq_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=CAMPOS)
                    w.writeheader()
                    w.writerows(linhas)
                json.dump(completo, open(arq_json, "w"), indent=1, default=float)
    return linhas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablacao", action="store_true")
    a = ap.parse_args()
    if a.ablacao:
        rodar(ABLACAO, ["ufg", "concavo_U"], [2, 3, 4, 5, 6], [True], RES / "ablacao.csv", RES / "ablacao.json")
    else:
        rodar(METODOS, list(CENARIOS), [2, 3, 4, 5, 6], [False, True], RES / "benchmark.csv", RES / "benchmark.json")
