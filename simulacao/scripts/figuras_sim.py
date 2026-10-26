"""Figuras do relatorio da simulacao (matplotlib, 150 dpi, legendas em portugues).

  python scripts/figuras_sim.py --cenario cenario/reduzido --execucao saidas/run01 \
      --contagens 40_m:resultados/contagem_run01.json 30_m:resultados/contagem_run02_a30.json \
      --vigia saidas/run01.vigia.log --saida evidencias --prefixo sim_

Gera: <prefixo>11_mapa_posse_dedup.png, 12_contagem_tempo.png, 13_erros_deteccao_contagem.png e,
com --vigia, 16_memoria_render.png.
"""
import argparse
import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MP  # noqa: E402

from cerise_sim.avaliacao import pessoas_vistas  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402

# paleta categorica de referencia (skill dataviz) e status
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BOM, AVISO, SERIO, CRITICO = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
TINTA, TINTA2, GRADE, SUPERF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NOMES = {"botsort": "BoT-SORT recomendado (0,40, fuse_score)", "botsort_cal": "BoT-SORT calibrado (0,15, sem fuse_score)",
         "botsort_pose": "BoT-SORT (GMC pela pose)", "chao": "rastreador no solo (conf ≥ 0,175)"}
plt.rcParams.update({"figure.facecolor": SUPERF, "axes.facecolor": SUPERF, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})


def ler(rotulo_caminho):
    r, c = rotulo_caminho.split(":", 1)
    return r.replace("_", " "), json.loads(Path(c).read_text(encoding="utf-8"))


def mapa(cen, ex, cont, caminho, modo=None):
    modo = modo or cont["principal"]
    d = cont[modo]
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    ids = list(cen["regioes_enu"])
    for i, did in enumerate(ids):
        for pl in cen["regioes_enu"][did]:
            ax.add_patch(MP(pl, closed=True, fc=CAT[i], alpha=0.10, ec=CAT[i], lw=1.0))
    for fr in cen.get("fronteiras_enu", []) or []:
        try:
            a = np.asarray(fr, float)
            if a.ndim == 2:
                ax.plot(a[:, 0], a[:, 1], color=TINTA2, lw=0.8, ls=":")
        except Exception:  # noqa: BLE001
            pass
    for i, did in enumerate(ids):
        if did not in ex.cameras("drone"):
            continue
        qs = ex.quadros(did)
        xy = np.array([q.pose.C[:2] for q in qs]); cob = np.array([q.cobre for q in qs])
        ax.plot(xy[:, 0], xy[:, 1], color=CAT[i], lw=0.6, alpha=0.45)
        xc = np.where(cob[:, None], xy, np.nan)
        ax.plot(xc[:, 0], xc[:, 1], color=CAT[i], lw=1.4, label=f"{did}: rota (cobertura em traço cheio)")
        ax.annotate(did, xy[0], color=TINTA, fontsize=9, fontweight="bold", xytext=(4, 4), textcoords="offset points")
    # clusters: 1o de cada rotulo = correto, demais = duplicata
    cls = sorted(d["clusters"], key=lambda c: c["t_ini"])
    vistos = set()
    pts = {"correto": [], "dup": [], "falso": []}
    for c in cls:
        if c["rotulo"] is None:
            pts["falso"].append(c["xy"])
        elif c["rotulo"] in vistos:
            pts["dup"].append(c["xy"])
        else:
            pts["correto"].append(c["xy"]); vistos.add(c["rotulo"])
    gab = ex.gabarito()
    usados = {cam: set() for cam in d["faixas_por_drone"]}
    for cam in usados:
        usados[cam] = {q.frame for q in ex.quadros(cam) if q.cobre}
    vis = pessoas_vistas({c: ex.caixas_gt(c) for c in usados}, usados)
    perdidas = vis - vistos
    ult = {}
    for f in sorted(gab):
        for pid, p in gab[f].items():
            ult[pid] = p
    pp = np.array([ult[p] for p in perdidas if p in ult]).reshape(-1, 2)
    av = d["avaliacao_contagem"]
    for k, (mk, cor, nome, mfc) in {"correto": ("o", BOM, "pessoa contada (cluster correto)", "none"),
                                    "dup": ("s", SERIO, "duplicata", SERIO), "falso": ("X", CRITICO, "falso positivo", CRITICO)}.items():
        a = np.array(pts[k]).reshape(-1, 2)
        if len(a):
            ax.plot(a[:, 0], a[:, 1], mk, mfc=mfc, mec=cor, ms=6.5, mew=1.3, ls="none", label=f"{nome} ({len(a)})")
    if len(pp):
        ax.plot(pp[:, 0], pp[:, 1], "+", color=TINTA, ms=8, mew=1.4, ls="none", label=f"pessoa vista e não contada ({len(pp)}; posição final)")
    ax.set_aspect("equal")
    ax.set_xlabel("x leste (m, ENU)"); ax.set_ylabel("y norte (m, ENU)")
    ax.set_title(f"Posse das subáreas, rotas e contagem única - {NOMES.get(modo, modo)}" + chr(10) +
                 f"{av['contagem_proposta']} estimadas × {av['pessoas_vistas_gabarito']} pessoas vistas (cena: {av['pessoas_na_cena']})",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def contagem_tempo(conts, caminho, modo=None):
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    ax = axs[0]
    for i, (rot, c) in enumerate(conts):
        st = c[modo or c["principal"]]["serie_temporal"]
        ax.plot(st["t"], st["estimada_acumulada"], color=CAT[i], lw=2, label=f"{rot}: contagem única estimada")
        ax.plot(st["t"], st["vistas_acumulada_gabarito"], color=CAT[i], lw=1.2, ls="--", label=f"{rot}: pessoas distintas já vistas (gabarito)")
    n = conts[0][1][modo or conts[0][1]["principal"]]["avaliacao_contagem"]["pessoas_na_cena"]
    ax.axhline(n, color=TINTA2, lw=0.8, ls=":")
    ax.text(1, n + 1, f"pessoas na cena = {n}", color=TINTA2, fontsize=8)
    ax.set_ylabel("pessoas (acumulado)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"Contagem acumulada ao longo da missão - {NOMES.get(modo, modo)}" + chr(10) +
                 "(agregação offline: cada cluster entra no instante da sua 1ª observação)", fontsize=9, loc="left")
    ax = axs[1]
    rot, c = conts[0]
    pdz = c[modo or c["principal"]]["serie_temporal"]["por_drone"]
    for i, (cam, s) in enumerate(pdz.items()):
        t = np.array(s["t"], float); e = np.array(s["estimada"], float); g = np.array(s["gabarito"], float)
        if not len(t):
            continue
        quebra = np.nonzero(np.diff(t) > 0.5)[0] + 1   # nao liga trechos de cobertura separados por transito
        t = np.insert(t, quebra, np.nan); e = np.insert(e, quebra, np.nan); g = np.insert(g, quebra, np.nan)
        ax.plot(t, g, color=CAT[i], lw=0.9, ls="--", alpha=0.9)
        ax.plot(t, e, color=CAT[i], lw=1.4, label=f"{cam}")
    ax.plot([], [], color=TINTA2, lw=1.4, label="traço cheio: trilhas no quadro")
    ax.plot([], [], color=TINTA2, lw=0.9, ls="--", label="tracejado: caixas de gabarito")
    ax.set_xlabel("tempo de missão (s)"); ax.set_ylabel(f"pessoas no quadro ({rot})")
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax.set_title("Contagem atual por drone (quadros de cobertura)", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def erros(conts, caminho, modo_trilhas=None):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axs[0]
    crit = [("iou0.5", "IoU ≥ 0,5"), ("iou0.3", "IoU ≥ 0,3"), ("iou0.3_ou_centro6px", "IoU ≥ 0,3 ou centro ≤ 6 px")]
    larg = 0.8 / max(len(conts), 1)
    for i, (rot, c) in enumerate(conts):
        apv = [c.get("deteccao_bruta", {}).get("ap", {}).get(k, {}).get("ap", np.nan) for k, _ in crit]
        tr = c[modo_trilhas if modo_trilhas in c else c["principal"]]["deteccao_por_quadro_trilhas"]
        P = [tr[k]["precisao"] for k, _ in crit]; R = [tr[k]["revocacao"] for k, _ in crit]
        x = np.arange(3) * 3
        for j, (vals, nome) in enumerate(((apv, "AP (detecções)"), (P, "precisão (trilhas)"), (R, "revocação (trilhas)"))):
            xx = x + j + (i - (len(conts) - 1) / 2) * larg
            ax.bar(xx, vals, width=larg * 0.92, color=CAT[j], alpha=1.0 if i == 0 else 0.55,
                   label=f"{nome}" if i == 0 else None, edgecolor=SUPERF, linewidth=1)
            for xv, v in zip(xx, vals):
                if np.isfinite(v):
                    ax.text(xv, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5, color=TINTA2, rotation=90)
    ax.set_xticks(np.arange(3) * 3 + 1); ax.set_xticklabels([n for _, n in crit], fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel("valor")
    sub = " | ".join(f"{rot}: {'cheio' if i == 0 else 'claro'}" for i, (rot, _) in enumerate(conts))
    ax.set_title(f"Detecção por quadro x gabarito do Replicator; trilhas: {NOMES.get(modo_trilhas, modo_trilhas)}" + chr(10) + f"({sub})",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", ncol=3)
    ax = axs[1]
    linhas = []
    for rot, c in conts:
        for modo in [m for m in ("botsort", "botsort_cal", "chao") if m in c]:
            av = c[modo]["avaliacao_contagem"]
            linhas.append((f"{rot}: {NOMES[modo]}", av))
    y = np.arange(len(linhas))
    for k, (rot, av) in enumerate(linhas):
        c1, c2, c3 = av["clusters_corretos_unicos"], av["clusters_duplicados"], av["clusters_falsos"]
        ax.barh(k, c1, color=BOM, height=0.6, label="corretos" if k == 0 else None, edgecolor=SUPERF)
        ax.barh(k, c2, left=c1, color=SERIO, height=0.6, label="duplicatas" if k == 0 else None, edgecolor=SUPERF)
        ax.barh(k, c3, left=c1 + c2, color=CRITICO, height=0.6, label="falsos" if k == 0 else None, edgecolor=SUPERF)
        ax.plot(av["pessoas_vistas_gabarito"], k, "|", color=TINTA, ms=16, mew=2, label="pessoas vistas (gabarito)" if k == 0 else None)
        ax.text(max(c1 + c2 + c3, av["pessoas_vistas_gabarito"]) * 1.02 + 0.5, k, f"{av['contagem_proposta']} ({av['erro_pct_vs_vistas']:+.1f}%); perdidas {av['pessoas_perdidas']}",
                va="center", fontsize=7.5, color=TINTA)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _ in linhas], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("clusters da contagem única")
    ax.set_xlim(0, max(av["contagem_proposta"] for _, av in linhas) * 1.45)
    ax.set_title("Contagem única global: decomposição contra o gabarito", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=4)
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def memoria(vigia, ex, caminho):
    import csv
    t, av, cm = [], [], []
    with open(vigia, encoding="utf-8") as f:
        for ln in f:
            p = ln.strip().split(",")
            if len(p) >= 4 and p[1].isdigit():
                h, m, s = map(int, p[0].split(":"))
                t.append(h * 3600 + m * 60 + s); av.append(int(p[1]) / 1024)
                v = p[3]
                try:
                    cm.append(float(v[:-3]) * (1 if v.endswith("GiB") else 1 / 1024) if v.endswith(("GiB", "MiB")) else np.nan)
                except ValueError:
                    cm.append(np.nan)
    t = (np.array(t) - t[0]) / 60
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, av, color=CAT[0], lw=1.6, label="memória 'available' do Spark (GB)")
    ax.plot(t, cm, color=CAT[1], lw=1.6, label="memória do contêiner Isaac (GB, docker stats)")
    ax.axhline(10, color=CRITICO, lw=1, ls="--")
    ax.text(0.3, 10.6, "limite do vigia (10 GB)", color=CRITICO, fontsize=8, ha="left")
    ax.set_xlabel("minutos desde o início do contêiner"); ax.set_ylabel("GB")
    m = ex.meta
    ax.set_title(f"Execução do Isaac Sim ({m.get('cenario')}, {m.get('quadros')} instantes x {len(m.get('cameras', {}))} câmeras): "
                 f"{m.get('s_por_quadro', 0):.2f} s por instante capturado".replace(".", ","), fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def sensibilidade(varrs, caminho):
    """Contagem unica x limiar (BoT-SORT com e sem fuse_score; rastreador no solo x conf), por altitude."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, (rot, v) in enumerate(varrs):
        b = v["botsort"]
        for fuse, ls, nome in ((True, "--", "com fuse_score (padrão)"), (False, "-", "sem fuse_score")):
            pts = sorted((r["track_high_new"], r["contagem_proposta"]) for r in b if r.get("fuse_score", True) == fuse)
            if pts:
                axs[0].plot(*zip(*pts), ls=ls, marker="o", ms=4, color=CAT[i], lw=1.6, label=f"{rot}: BoT-SORT {nome}")
        pts = sorted((r["conf"], r["contagem_proposta"]) for r in v["chao"])
        axs[1].plot(*zip(*pts), marker="s", ms=4, color=CAT[i], lw=1.6, label=f"{rot}: rastreador no solo")
        vis = b[0]["pessoas_vistas_gabarito"]
        for ax in axs:
            ax.axhline(vis, color=CAT[i], lw=0.8, ls=":")
            ax.text(0.80, vis + (2.0 if i == 0 else -7.0), f"{rot}: {vis} pessoas vistas", color=CAT[i], fontsize=7.5,
                    transform=ax.get_yaxis_transform(), ha="center")
    axs[0].axvline(0.40, color=TINTA2, lw=0.8); axs[0].text(0.405, 0.62, "recomendado" + chr(10) + "(vídeos reais)", fontsize=7, color=TINTA2,
                                                             transform=axs[0].get_xaxis_transform(), va="top")
    axs[0].axvline(0.15, color=TINTA2, lw=0.8, ls="--"); axs[0].text(0.155, 0.80, "calibrado" + chr(10) + "(voo a 40 m)", fontsize=7, color=TINTA2,
                                                                      transform=axs[0].get_xaxis_transform(), va="top")
    axs[1].axvline(0.175, color=TINTA2, lw=0.8); axs[1].text(0.18, 0.97, "fixado no" + chr(10) + "preparo", fontsize=7, color=TINTA2,
                                                              transform=axs[1].get_xaxis_transform(), va="top")
    axs[0].set_xlabel("track_high_thresh = new_track_thresh (BoT-SORT)"); axs[1].set_xlabel("confiança mínima da detecção (rastreador no solo)")
    axs[0].set_ylabel("contagem única estimada")
    axs[0].set_title("BoT-SORT + deduplicação no solo", fontsize=9.5, loc="left")
    axs[1].set_title("Rastreador próprio no solo + deduplicação", fontsize=9.5, loc="left")
    for ax in axs:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--contagens", nargs="+", required=True, help="rotulo:caminho.json (o 1o e o principal; '_' no rotulo vira espaco)")
    ap.add_argument("--vigia", default=None)
    ap.add_argument("--varreduras", nargs="*", default=[], help="rotulo:varredura.json (scripts/varredura_botsort.py)")
    ap.add_argument("--saida", default="evidencias")
    ap.add_argument("--prefixo", default="sim_")
    ap.add_argument("--modo", default="botsort_cal", help="rastreador do mapa, da serie temporal e das metricas de trilha")
    a = ap.parse_args()
    cen = json.loads((Path(a.cenario) / "cenario.json").read_text(encoding="utf-8"))
    ex = Execucao(a.execucao)
    conts = [ler(x) for x in a.contagens]
    out = Path(a.saida); out.mkdir(parents=True, exist_ok=True)
    modo = a.modo if a.modo in conts[0][1] else conts[0][1]["principal"]
    mapa(cen, ex, conts[0][1], out / f"{a.prefixo}11_mapa_posse_dedup.png", modo)
    contagem_tempo(conts, out / f"{a.prefixo}12_contagem_tempo.png", modo)
    erros(conts, out / f"{a.prefixo}13_erros_deteccao_contagem.png", modo)
    if a.varreduras:
        sensibilidade([ler(x) for x in a.varreduras], out / f"{a.prefixo}15_sensibilidade_rastreador.png")
    if a.vigia:
        memoria(a.vigia, ex, out / f"{a.prefixo}16_memoria_render.png")
    print("figuras em", out)


if __name__ == "__main__":
    main()
