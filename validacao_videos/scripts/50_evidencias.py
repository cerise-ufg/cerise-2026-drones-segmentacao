"""Etapa 5 (CPU): figuras, tabelas e quadros dos videos de evidencia a partir dos resultados medidos.

Combinacao principal = maior HOTA no VAL entre as variantes do detector ajustado; referencia = maior HOTA no
VAL entre as variantes COCO (as duas escolhidas antes de olhar o test-dev).
Saidas: evidencias/v01..v07*.png, resultados/tabela_por_sequencia.csv, resultados/condicoes_analise.json,
video_tmp/<nome>/*.jpg (quadros para o ffmpeg do host montar os MP4 H.264).
"""
import csv
import glob
import json
import os
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import metricas_mot as mm

BASE = "/w/validacao_videos"; RES = f"{BASE}/resultados"; EV = f"{BASE}/evidencias"; MOT = "/w/mot"
os.makedirs(EV, exist_ok=True)
sel = json.load(open(f"{RES}/selecao_val.json")); ag = json.load(open(f"{RES}/metricas_agregadas.json"))
ps = json.load(open(f"{RES}/metricas_por_sequencia.json")); cdet = json.load(open(f"{RES}/contagem_deteccoes.json"))
cond = {(c["split"], c["seq"]): c for c in json.load(open(f"{RES}/condicoes_sequencias.json"))}
esc = sel["escolha"]
PRINC = max([k for k in esc if k.startswith("ft")], key=lambda k: esc[k]["HOTA_val"])
REF = max([k for k in esc if k.startswith("coco")], key=lambda k: esc[k]["HOTA_val"])
NOMES = {"ft960": "YOLO11s ajustado 960", "ft1280": "YOLO11s ajustado 1280", "coco960": "YOLO11s COCO 960",
         "coco1280": "YOLO11s COCO 1280", "bytetrack": "ByteTrack", "botsort": "BoT-SORT"}


def nome(combo):
    v, n = combo.split("|")
    return f"{NOMES[v]} + {NOMES[n]}"


CURTO = {"ft960": "ajust. 960", "ft1280": "ajust. 1280", "coco960": "COCO 960", "coco1280": "COCO 1280"}


def curto(combo):
    v, n = combo.split("|")
    return CURTO[v] + "\n" + NOMES[n]


print("principal:", PRINC, "referencia:", REF, flush=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
FONTE = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans.ttf")
FONTE_B = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")


# ------------------------------------------------------------------ series por quadro
def serie(split, seq, combo):
    v, n = combo.split("|")
    T = len(os.listdir(f"{MOT}/VisDrone2019-MOT-{split}/sequences/{seq}"))
    g, ign = mm.ler_gabarito(f"{MOT}/VisDrone2019-MOT-{split}/annotations/{seq}.txt", T)
    gt = {t: mm.filtrar_ignorado(g[t][0], g[t][1], ign[t]) for t in range(1, T + 1)}
    d = np.loadtxt(f"{BASE}/rastreios/{v}_{n}/{split}/{seq}.txt", delimiter=",", ndmin=2)
    trk = {}
    for t in range(1, T + 1):
        x = d[d[:, 0] == t] if len(d) else np.zeros((0, 10))
        trk[t] = (x[:, 1].astype(int), np.c_[x[:, 2], x[:, 3], x[:, 2] + x[:, 4], x[:, 3] + x[:, 5]])
    z = np.load(f"{BASE}/cache_det/{v}/{split}/{seq}.npz")
    tau = cdet[v]["tau_val"]
    cont_det = np.zeros(T, int)
    for t in range(1, T + 1):
        m = (z["quadro"] == t) & (z["conf"] >= tau)
        xy = z["xyxy"][m]
        cont_det[t - 1] = int(np.sum(mm.fracao_em_ignorado(xy, ign[t]) <= 0.5))
    L = esc[combo]["L"]
    vis_gt, vis_trk, comp = set(), set(), {}
    s = {k: np.zeros(T, int) for k in ("gt", "trk", "det", "acum_gt", "acum_trk_L1", "acum_trk_L")}
    nL = 0
    for t in range(1, T + 1):
        s["gt"][t - 1] = len(gt[t][0]); s["trk"][t - 1] = len(trk[t][0]); s["det"][t - 1] = cont_det[t - 1]
        vis_gt.update(gt[t][0].tolist()); vis_trk.update(trk[t][0].tolist())
        for i in trk[t][0].tolist():
            comp[i] = comp.get(i, 0) + 1
            if comp[i] == L:
                nL += 1  # contagem causal: o ID entra no acumulado quando completa L quadros
        s["acum_gt"][t - 1] = len(vis_gt); s["acum_trk_L1"][t - 1] = len(vis_trk); s["acum_trk_L"][t - 1] = nL
    return {"T": T, "L": L, "tau": tau, "gt_q": gt, "ign_q": ign, "trk_q": trk, "comp": comp, **s}


# ------------------------------------------------------------------ v01 contagem x tempo
SEQS_FIG = ["uav0000073_04464_v", "uav0000120_04775_v", "uav0000077_00720_v"]
ROT = {"uav0000073_04464_v": "rua comercial ao entardecer, densa", "uav0000120_04775_v": "cruzamento à noite",
       "uav0000077_00720_v": "pátio com caminhões, esparsa"}
fig, axs = plt.subplots(len(SEQS_FIG), 2, figsize=(12, 3.0 * len(SEQS_FIG)))
series = {}
for i, seq in enumerate(SEQS_FIG):
    sp = serie("test-dev", seq, PRINC); sr = serie("test-dev", seq, REF)
    series[seq] = sp
    t = np.arange(1, sp["T"] + 1)
    a = axs[i, 0]
    a.plot(t, sp["gt"], "k-", lw=1.6, label="gabarito (pessoas anotadas)")
    a.plot(t, sp["trk"], "-", color="#1f77b4", lw=1.1, label=f"rastreadas — {nome(PRINC)}")
    a.plot(t, sp["det"], "-", color="#ff7f0e", lw=0.8, alpha=0.8, label=f"detecções (conf ≥ {sp['tau']:.3f}) — {NOMES[PRINC.split('|')[0]]}")
    a.plot(t, sr["trk"], "--", color="0.5", lw=0.9, label=f"rastreadas — {nome(REF)}")
    a.set_title(f"{seq} ({ROT[seq]}): pessoas por quadro"); a.set_xlabel("quadro"); a.set_ylabel("pessoas")
    a.legend(fontsize=6.5, loc="best")
    b = axs[i, 1]
    b.plot(t, sp["acum_gt"], "k-", lw=1.6, label="gabarito (IDs únicos)")
    b.plot(t, sp["acum_trk_L1"], ":", color="#1f77b4", lw=1.3, label="rastreador, todos os IDs (L=1)")
    b.plot(t, sp["acum_trk_L"], "-", color="#1f77b4", lw=1.3, label=f"rastreador, IDs com ≥ {sp['L']} quadros")
    b.plot(t, sr["acum_trk_L"], "--", color="0.5", lw=1.0, label=f"referência COCO, ≥ {sr['L']} quadros")
    b.set_title("pessoas únicas acumuladas"); b.set_xlabel("quadro"); b.set_ylabel("pessoas únicas"); b.legend(fontsize=6.5)
fig.suptitle("Contagem detectada × gabarito no tempo (VisDrone2019-MOT test-dev, parâmetros escolhidos no val)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{EV}/v01_contagem_tempo.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ v02 metricas agregadas
combos = [f"{v}|{n}" for v in ("ft960", "ft1280", "coco960", "coco1280") for n in ("bytetrack", "botsort")]
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
x = np.arange(len(combos)); w = 0.27
for j, (met, cor) in enumerate((("HOTA", "#1f77b4"), ("IDF1", "#2ca02c"), ("MOTA", "#d62728"))):
    vals = [ag[f"{c}|escolhida_val|test-dev"][met] for c in combos]
    pad = [ag[f"{c}|padrao|test-dev"][met] for c in combos]
    axs[0].bar(x + (j - 1) * w, vals, w, color=cor, label=f"{met} (config. do val)")
    axs[0].scatter(x + (j - 1) * w, pad, marker="_", s=120, color="k", zorder=3, label="padrão ultralytics" if j == 0 else None)
axs[0].set_xticks(x); axs[0].set_xticklabels([curto(c) for c in combos], fontsize=7)
axs[0].set_title("Rastreamento de pessoas (test-dev)"); axs[0].set_ylim(-0.08, 0.55); axs[0].legend(fontsize=7, ncol=2, loc="upper left"); axs[0].axhline(0, color="k", lw=0.6)
mae_trk = [ag[f"{c}|escolhida_val|test-dev"]["contagem_quadro_trk"]["MAE"] for c in combos]
mae_det = [cdet[c.split("|")[0]]["test-dev"]["_agregado"]["MAE"] for c in combos]
axs[1].bar(x - 0.2, mae_trk, 0.4, color="#1f77b4", label="contagem pelas trilhas ativas")
axs[1].bar(x + 0.2, mae_det, 0.4, color="#ff7f0e", label="contagem pelas detecções (limiar do val)")
axs[1].set_xticks(x); axs[1].set_xticklabels([curto(c) for c in combos], fontsize=7)
axs[1].set_ylabel("MAE (pessoas/quadro)"); axs[1].set_title("Erro de contagem por quadro (test-dev)"); axs[1].legend(fontsize=7)
e1 = [100 * ag[f"{c}|escolhida_val|test-dev"]["unicos"]["erro_rel_abs_medio_L1"] for c in combos]
eL = [100 * ag[f"{c}|escolhida_val|test-dev"]["unicos"]["erro_rel_abs_medio_L"] for c in combos]
axs[2].bar(x - 0.2, e1, 0.4, color="#9467bd", label="todos os IDs (L=1)")
axs[2].bar(x + 0.2, eL, 0.4, color="#8c564b", label="IDs com ≥ L quadros (L do val)")
axs[2].set_xticks(x); axs[2].set_xticklabels([curto(c) for c in combos], fontsize=7)
axs[2].set_ylabel("erro relativo absoluto médio (%)"); axs[2].set_title("Contagem de pessoas únicas por sequência (test-dev)")
axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{EV}/v02_metricas_agregadas.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ v03 contagem unica por sequencia + decomposicao
lin = {(r["variante"] + "|" + r["rastreador"], r["split"], r["seq"]): r for r in ps if r["rotulo"] == "escolhida_val"}
seqs_td = sorted({r["seq"] for r in ps if r["split"] == "test-dev"})
seqs_td = [s for s in seqs_td if lin[(PRINC, "test-dev", s)]["unicos_gt"] > 0]
fig, axs = plt.subplots(1, 2, figsize=(16.5, 4.8), gridspec_kw={"width_ratios": [2.0, 1.1]})
x = np.arange(len(seqs_td))
gtv = [lin[(PRINC, "test-dev", s)]["unicos_gt"] for s in seqs_td]
p1 = [lin[(PRINC, "test-dev", s)]["unicos_pred"]["1"] for s in seqs_td]
pL = [lin[(PRINC, "test-dev", s)]["unicos_pred_L"] for s in seqs_td]
rL = [lin[(REF, "test-dev", s)]["unicos_pred_L"] for s in seqs_td]
axs[0].bar(x - 0.3, gtv, 0.2, color="k", label="gabarito")
axs[0].bar(x - 0.1, p1, 0.2, color="#9467bd", label=f"{nome(PRINC)}, L=1")
axs[0].bar(x + 0.1, pL, 0.2, color="#1f77b4", label=f"{nome(PRINC)}, L={esc[PRINC]['L']}")
axs[0].bar(x + 0.3, rL, 0.2, color="0.6", label=f"{nome(REF)}, L={esc[REF]['L']}")
axs[0].set_yscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels([s[3:13] for s in seqs_td], rotation=60, fontsize=7)
axs[0].set_ylabel("pessoas únicas (escala log)"); axs[0].legend(fontsize=7); axs[0].set_title("Pessoas únicas por sequência (test-dev)")
dec_keys = ("trk_ids_falsos", "excesso_troca_fragmentacao", "gt_ids_perdidos")
xx = np.arange(len(combos))
dd = {k: [ag[f"{c}|escolhida_val|test-dev"]["unicos"]["decomposicao_L1"][k] for c in combos] for k in dec_keys}
gt_tot = ag[f"{PRINC}|escolhida_val|test-dev"]["unicos"]["gt_total"]
axs[1].bar(xx, dd["trk_ids_falsos"], color="#d62728", label="IDs falsos (nunca casados com pessoa)")
axs[1].bar(xx, dd["excesso_troca_fragmentacao"], bottom=dd["trk_ids_falsos"], color="#ff7f0e", label="IDs extras por troca/fragmentação")
axs[1].bar(xx, [-v for v in dd["gt_ids_perdidos"]], color="#7f7f7f", label="pessoas nunca rastreadas (−)")
axs[1].scatter(xx, [ag[f"{c}|escolhida_val|test-dev"]["unicos"]["pred_total_L1"] - gt_tot for c in combos], color="k", marker="D",
               zorder=3, label="inflação líquida (IDs − gabarito)")
axs[1].axhline(0, color="k", lw=0.7)
axs[1].set_xticks(xx); axs[1].set_xticklabels([curto(c) for c in combos], fontsize=6.5)
axs[1].set_title(f"Decomposição da contagem bruta de IDs (L=1), test-dev\n(gabarito: {gt_tot} pessoas)")
axs[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{EV}/v03_contagem_unica.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ tabela por sequencia
campos = ["split", "seq", "resolucao", "luminancia", "pessoas_por_quadro_gt", "altura_pessoa_px", "mov_camera_pct", "combo",
          "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "MAE_quadro_trk", "MAPE_quadro_trk_pct", "vies_quadro_trk",
          "MAE_quadro_det", "MAPE_quadro_det_pct", "vies_quadro_det", "unicos_gt", "unicos_pred_L1", "unicos_pred_L", "L",
          "erro_rel_unico_L1_pct", "erro_rel_unico_L_pct", "ids_falsos", "ids_extras_troca", "pessoas_perdidas"]
tab = []
for combo in (PRINC, REF):
    for split in ("val", "test-dev"):
        for s in sorted({r["seq"] for r in ps if r["split"] == split}):
            r = lin[(combo, split, s)]; c = cond[(split, s)]
            cd = cdet[combo.split("|")[0]][split][s]
            g = r["unicos_gt"]
            tab.append({"split": split, "seq": s, "resolucao": f"{c['largura']}x{c['altura']}", "luminancia": round(c["luminancia_media"], 1),
                        "pessoas_por_quadro_gt": round(c["pessoas_por_quadro_media"], 2),
                        "altura_pessoa_px": round(c["altura_pessoa_px_mediana"], 1) if g else "", "mov_camera_pct": round(c["mov_camera_pct_largura_por_quadro_mediana"], 3),
                        "combo": combo, "HOTA": round(r["HOTA"], 4), "DetA": round(r["DetA"], 4), "AssA": round(r["AssA"], 4),
                        "MOTA": round(r["MOTA"], 4), "IDF1": round(r["IDF1"], 4), "IDSW": r["IDSW"],
                        "MAE_quadro_trk": round(r["contagem_quadro_trk"]["MAE"], 3), "MAPE_quadro_trk_pct": round(r["contagem_quadro_trk"]["MAPE_pct"], 2),
                        "vies_quadro_trk": round(r["contagem_quadro_trk"]["vies_medio"], 3),
                        "MAE_quadro_det": round(cd["MAE"], 3), "MAPE_quadro_det_pct": round(cd["MAPE_pct"], 2), "vies_quadro_det": round(cd["vies_medio"], 3),
                        "unicos_gt": g, "unicos_pred_L1": r["unicos_pred"]["1"], "unicos_pred_L": r["unicos_pred_L"], "L": r["L_escolhido"],
                        "erro_rel_unico_L1_pct": round(100 * (r["unicos_pred"]["1"] - g) / g, 1) if g else "",
                        "erro_rel_unico_L_pct": round(100 * (r["unicos_pred_L"] - g) / g, 1) if g else "",
                        "ids_falsos": r["decomp_L1"]["trk_ids_falsos"], "ids_extras_troca": r["decomp_L1"]["excesso_troca_fragmentacao"],
                        "pessoas_perdidas": r["decomp_L1"]["gt_ids_perdidos"]})
with open(f"{RES}/tabela_por_sequencia.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=campos); wr.writeheader(); wr.writerows(tab)

# ------------------------------------------------------------------ v05 condicoes (test-dev, sequencias com pessoas)
td = [t for t in tab if t["combo"] == PRINC and t["split"] == "test-dev" and t["unicos_gt"] > 0]
noite = [t["luminancia"] < 70 for t in td]
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
for a, (campo, rot, logx) in zip(axs, (("pessoas_por_quadro_gt", "pessoas por quadro (gabarito, média)", True),
                                       ("altura_pessoa_px", "altura mediana da pessoa (px) — proxy de altitude", False),
                                       ("luminancia", "luminância média do quadro (0–255)", False),
                                       ("mov_camera_pct", "movimento de câmera (% da largura/quadro)", False))):
    xs = [t[campo] for t in td]
    a.scatter(xs, [t["HOTA"] for t in td], c=["#1f3b73" if n else "#f2a900" for n in noite], s=28, label="HOTA", zorder=3)
    a.scatter(xs, [t["MAPE_quadro_trk_pct"] / 100 for t in td], marker="x", c="#d62728", s=22, label="MAPE por quadro (÷100)")
    a.set_xlabel(rot, fontsize=8)
    if logx:
        a.set_xscale("log")
axs[0].set_ylabel("métrica"); axs[0].legend(fontsize=7)
axs[2].text(0.02, 0.97, "azul = noite (lum. < 70)\namarelo = dia/entardecer", transform=axs[2].transAxes, va="top", fontsize=7)
fig.suptitle(f"Desempenho por condição — {nome(PRINC)}, 16 sequências do test-dev com pessoas", fontsize=10)
fig.tight_layout(); fig.savefig(f"{EV}/v05_condicoes.png", dpi=150); plt.close(fig)


def grupos(lst, chave):
    out = {}
    for t in lst:
        out.setdefault(chave(t), []).append(t)
    return {k: {"n_seq": len(v), "HOTA_media": float(np.mean([x["HOTA"] for x in v])), "IDF1_media": float(np.mean([x["IDF1"] for x in v])),
                "MOTA_media": float(np.mean([x["MOTA"] for x in v])), "MAPE_quadro_trk_media": float(np.mean([x["MAPE_quadro_trk_pct"] for x in v])),
                "MAE_quadro_trk_media": float(np.mean([x["MAE_quadro_trk"] for x in v])),
                "erro_rel_unico_L_medio_abs": float(np.mean([abs(x["erro_rel_unico_L_pct"]) for x in v])),
                "seqs": [x["seq"] for x in v]} for k, v in sorted(out.items())}


analise = {"combo": PRINC, "criterios": {"iluminacao": "noite se luminancia media < 70",
                                         "densidade": "baixa < 5, media 5-30, alta > 30 pessoas/quadro",
                                         "escala": "pequena < 35 px, media 35-60 px, grande > 60 px (altura mediana)",
                                         "camera": "em movimento se mediana do deslocamento GMC > 0,25% da largura/quadro"}}
for combo in (PRINC, REF):
    td_c = [t for t in tab if t["combo"] == combo and t["split"] == "test-dev" and t["unicos_gt"] > 0]
    analise[combo] = {
        "iluminacao": grupos(td_c, lambda t: "noite" if t["luminancia"] < 70 else "dia"),
        "densidade": grupos(td_c, lambda t: "1_baixa" if t["pessoas_por_quadro_gt"] < 5 else ("2_media" if t["pessoas_por_quadro_gt"] <= 30 else "3_alta")),
        "escala": grupos(td_c, lambda t: "1_pequena" if t["altura_pessoa_px"] < 35 else ("2_media" if t["altura_pessoa_px"] <= 60 else "3_grande")),
        "camera": grupos(td_c, lambda t: "em_movimento" if t["mov_camera_pct"] > 0.25 else "quase_parada"),
        "correlacao_spearman_HOTA": {}}
    from scipy.stats import spearmanr
    for campo in ("pessoas_por_quadro_gt", "altura_pessoa_px", "luminancia", "mov_camera_pct"):
        rho, p = spearmanr([t[campo] for t in td_c], [t["HOTA"] for t in td_c])
        analise[combo]["correlacao_spearman_HOTA"][campo] = {"rho": float(rho), "p": float(p)}
json.dump(analise, open(f"{RES}/condicoes_analise.json", "w"), indent=1)

# ------------------------------------------------------------------ v06 grade no val
rows = list(csv.DictReader(open(f"{RES}/grade_val.csv")))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for a, n in zip(axs, ("bytetrack", "botsort")):
    for v, cor in zip(("ft960", "ft1280", "coco960", "coco1280"), ("#1f77b4", "#17becf", "#d62728", "#ff9896")):
        for b, ls in ((30, "-"), (60, "--")):
            rr = sorted([r for r in rows if r["variante"] == v and r["rastreador"] == n and float(r["track_low_thresh"]) == 0.05
                         and int(r["track_buffer"]) == b], key=lambda r: float(r["track_high_thresh"]))
            a.plot([float(r["track_high_thresh"]) for r in rr], [float(r["HOTA"]) for r in rr], ls, color=cor, marker=".",
                   label=f"{NOMES[v]}, buffer {b}")
    a.axvline(0.25, color="k", lw=0.6, ls=":")
    a.set_title(f"{NOMES[n]} — HOTA no val (track_low_thresh = 0,05)"); a.set_xlabel("track_high_thresh = new_track_thresh")
axs[0].set_ylabel("HOTA (val)"); axs[1].legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{EV}/v06_grade_val.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ v07 FPS
if os.path.exists(f"{RES}/fps_gb10.json"):
    fp = json.load(open(f"{RES}/fps_gb10.json"))["medidas"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for a, sq in zip(axs, ("4K (3840x2160)", "1344x756")):
        mm_ = [m for m in fp if m["sequencia"] == sq]
        rot = [curto(f"{m['detector']}|{m['rastreador']}") for m in mm_]
        xl = np.arange(len(mm_))
        b1 = [m["ms_leitura_jpeg"] for m in mm_]; b2 = [m["ms_deteccao"] for m in mm_]; b3 = [m["ms_rastreador"] for m in mm_]
        a.bar(xl, b1, color="#7f7f7f", label="leitura/decodificação JPEG")
        a.bar(xl, b2, bottom=b1, color="#1f77b4", label="detecção YOLO11s (FP32)")
        a.bar(xl, b3, bottom=np.add(b1, b2), color="#ff7f0e", label="rastreador (inclui GMC no BoT-SORT)")
        for i, m in enumerate(mm_):
            a.text(i, m["ms_total"] + 1, f"{m['fps_ponta_a_ponta']:.1f} FPS", ha="center", fontsize=7)
        a.set_xticks(xl); a.set_xticklabels(rot, fontsize=6.5); a.set_title(f"Pipeline em fluxo no GB10 — vídeo {sq}")
    axs[0].set_ylabel("ms por quadro"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{EV}/v07_fps_gb10.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ quadros com caixas/IDs e videos
def cor_id(i):
    h = (i * 47) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(v) for v in c)


def desenhar(split, seq, t, sp, largura=1280, gt_caixas=False, titulo=None):
    arq = sorted(glob.glob(f"{MOT}/VisDrone2019-MOT-{split}/sequences/{seq}/*.jpg"))[t - 1]
    img = cv2.imread(arq)[:, :, ::-1]
    s = largura / img.shape[1]
    img = cv2.resize(img, (largura, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img); dr = ImageDraw.Draw(pil, "RGBA")
    f_id = ImageFont.truetype(FONTE_B, max(10, int(11 * largura / 1280)))
    f_hud = ImageFont.truetype(FONTE, max(13, int(17 * largura / 1280)))
    if gt_caixas:
        for b in sp["ign_q"][t]:  # regioes ignoradas do VisDrone (classes 0 e 11): fora da avaliacao
            dr.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], fill=(128, 128, 128, 90), outline=(200, 200, 200, 200), width=1)
        for b in sp["gt_q"][t][1]:
            dr.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=(255, 255, 255, 170), width=1)
    for i, b in zip(sp["trk_q"][t][0], sp["trk_q"][t][1]):
        c = cor_id(int(i))
        dr.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=c + (255,), width=2)
        tx = str(int(i)); x0, y0 = b[0] * s, max(0, b[1] * s - f_id.size - 3)
        tw = dr.textlength(tx, font=f_id)
        dr.rectangle([x0, y0, x0 + tw + 4, y0 + f_id.size + 2], fill=c + (210,))
        dr.text((x0 + 2, y0), tx, font=f_id, fill=(0, 0, 0))
    acum_L = sp["acum_trk_L"][t - 1]; acum_1 = sp["acum_trk_L1"][t - 1]
    linhas = [titulo or f"VisDrone2019-MOT {split} {seq} — quadro {t}/{sp['T']}",
              f"{nome(PRINC)} | GB10, FP32",
              f"No quadro: {sp['trk'][t-1]} rastreadas  |  gabarito: {sp['gt'][t-1]}",
              f"Acumulado: {acum_L} pessoas únicas (IDs com ≥ {sp['L']} quadros; IDs brutos {acum_1})  |  gabarito: {sp['acum_gt'][t-1]}"]
    if gt_caixas:
        linhas.append("Coloridas = trilhas (número = ID); brancas finas = gabarito; cinza = região ignorada pelo VisDrone")
    hh = len(linhas) * (f_hud.size + 6) + 10
    dr.rectangle([0, 0, pil.width, hh], fill=(0, 0, 0, 150))
    for k, l in enumerate(linhas):
        dr.text((10, 6 + k * (f_hud.size + 6)), l, font=f_hud, fill=(255, 255, 255))
    return pil


QUADROS = [("uav0000073_04464_v", 200), ("uav0000120_04775_v", 600), ("uav0000077_00720_v", 400)]
for k, (seq, t) in enumerate(QUADROS):
    sp = series[seq] if seq in series else serie("test-dev", seq, PRINC)
    desenhar("test-dev", seq, t, sp, largura=1600, gt_caixas=True).save(f"{EV}/v04{'abc'[k]}_quadro_{seq}_{t}.png")

VIDEOS = [("uav0000073_04464_v", 1, 312), ("uav0000120_04775_v", 1, 450)]
for seq, t0, t1 in ([] if os.environ.get("SEM_VIDEO") else VIDEOS):
    sp = series[seq] if seq in series else serie("test-dev", seq, PRINC)
    pasta = f"{BASE}/video_tmp/{seq}"
    shutil.rmtree(pasta, ignore_errors=True); os.makedirs(pasta)
    for i, t in enumerate(range(t0, min(t1, sp["T"]) + 1)):
        desenhar("test-dev", seq, t, sp, largura=1280).save(f"{pasta}/{i:06d}.jpg", quality=92)
    print("quadros do video:", seq, flush=True)
json.dump({"principal": PRINC, "referencia": REF, "videos": VIDEOS, "quadros": QUADROS}, open(f"{RES}/evidencias_meta.json", "w"), indent=1)
print("FIM_EVIDENCIAS", flush=True)
