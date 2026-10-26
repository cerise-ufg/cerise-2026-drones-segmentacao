"""Tamanhos de quadro e tempo no ar das mensagens LoRa + custo de comunicacao das estrategias de
realocacao (a partir de resultados/colaboracao.csv).

Uso: python scripts/protocolo_analise.py
"""
import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cerise_drones import protocolo as P  # noqa: E402
from cerise_drones.visualizar import TINTA, estilo  # noqa: E402

RES = RAIZ / "resultados"; EV = RAIZ / "evidencias"
ex = P.exemplos()
linhas = []
for nome, q in ex.items():
    linhas.append({"mensagem": nome, "quadro_B": len(q), "payload_B": len(q) - 10,
                   **{f"SF{sf}_ms": round(P.tempo_no_ar_ms(len(q), sf), 1) for sf in (7, 9, 10, 12)},
                   "origem": "cerise_proto v1 (frente LoRa)" if P.decodificar(q)["tipo"] in
                   ("HEARTBEAT", "POS", "AREA_ASSIGN", "COVERAGE_DONE", "COUNT_REPORT", "ACK") else "extensao proposta"})
with open(RES / "mensagens_lora.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
md = ["| mensagem | quadro (B) | payload (B) | SF7 (ms) | SF9 (ms) | SF10 (ms) | SF12 (ms) | origem |", "|---|---|---|---|---|---|---|---|"]
md += [f"| {l['mensagem']} | {l['quadro_B']} | {l['payload_B']} | {l['SF7_ms']} | {l['SF9_ms']} | {l['SF10_ms']} | {l['SF12_ms']} | {l['origem']} |" for l in linhas]
(RES / "mensagens_lora.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

estilo()
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
x = np.arange(len(linhas))
for k, (sf, cor) in enumerate([(7, "#86b6ef"), (9, "#2a78d6"), (10, "#184f95")]):
    ax.bar(x + (k - 1) * 0.27, [l[f"SF{sf}_ms"] for l in linhas], 0.25, color=cor, label=f"SF{sf}, 125 kHz")
for i, l in enumerate(linhas):
    ax.text(i, -40, f"{l['quadro_B']} B", ha="center", va="top", fontsize=7, color=TINTA)
ax.set_xticks(x); ax.set_xticklabels([l["mensagem"] for l in linhas], rotation=25, ha="right", fontsize=7.5)
ax.axhline(400, color="#d03b3b", lw=1, ls="--"); ax.text(len(linhas) - 0.5, 410, "limite de ocupacao 400 ms", ha="right", fontsize=7, color="#d03b3b")
ax.set_ylabel("tempo no ar por quadro (ms)")
ax.set_ylim(-90, None)
ax.set_title("(a) Tempo no ar por mensagem (CR 4/5, preambulo 8, CRC)", loc="left", fontsize=9, color=TINTA)
ax.legend(fontsize=7.5, loc="upper left")
cb = RES / "colaboracao.csv"
if cb.exists():
    df = pd.read_csv(cb)
    g = df[df.estrategia != "nenhuma"].groupby("estrategia").agg(q=("quadros_lora", "mean"), qmax=("quadros_lora", "max"),
                                                                 ar=("tempo_no_ar_SF9_ms", "mean"), armax=("tempo_no_ar_SF9_ms", "max"))
    ax = axs[1]
    nomes = {"reparticao": "reparticao (DARP)", "leilao": "leilao (SSI)"}
    xs = np.arange(len(g))
    ax.bar(xs, g.ar / 1000, 0.5, color=["#1baf7a", "#eb6834"][:len(g)])
    for i, (k, r) in enumerate(g.iterrows()):
        ax.text(i, r.ar / 1000 + 0.15, f"{r.q:.0f} quadros (max {r.qmax:.0f})\n{r.ar / 1000:.1f} s no ar (max {r.armax / 1000:.1f} s)",
                ha="center", fontsize=7.5, color=TINTA)
    ax.set_xticks(xs); ax.set_xticklabels([nomes.get(k, k) for k in g.index])
    ax.set_ylabel("tempo no ar total, SF9 (s)")
    ax.set_ylim(0, g.armax.max() / 1000 * 0.9)
    ax.set_title("(b) Custo de comunicacao da realocacao (media de 12 falhas)", loc="left", fontsize=9, color=TINTA)
fig.tight_layout()
fig.savefig(EV / "08_lora_mensagens.png", bbox_inches="tight")
plt.close(fig)
