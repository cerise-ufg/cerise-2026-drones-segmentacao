"""Pipeline de contagem sobre uma execucao gravada (Isaac Sim, Gazebo ou sintetica sem render).

Rastreadores (--rastreadores, lista; o 1o e o principal):
  botsort (padrao, configuracao recomendada pela validacao em videos reais): YOLO11s ajustado (FP32) com
          deteccoes em cache (conf >= 0,01, NMS 0,6) + BoT-SORT do ultralytics com GMC sparseOptFlow
          (track_high = new = 0,40, low = 0,05, buffer 30), trilhas projetadas no solo -> posse + dedup.
  chao    : rastreador proprio no plano do chao (Hungaro + alfa-beta) sobre deteccoes com conf >= --conf_chao.
  botsort_pose / botsort_semgmc: BoT-SORT com a GMC calculada pela telemetria (pose) ou sem GMC (ablacao).
  botsort_cal: BoT-SORT com --bs_cal (padrao: high=new=0,15, low=0,05, SEM fuse_score), escolhido a posteriori no
          voo a 40 m (scripts/varredura_botsort.py) e aplicado sem alteracao aos demais voos.
  Todos rodam sobre o mesmo cache de deteccoes.
A contagem unica e SEMPRE pela posicao no solo (posse da subarea + deduplicacao espaco-temporal), nao por IDs.

YOLO no conteiner da frente de drones (ver scripts/rodar_contagem_spark.sh):
  python scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido --detector yolo \
      --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 --saida resultados/contagem_run01.json --figura
Detector oraculo (gabarito + ruido; isola o erro da agregacao), rastreador no chao:
  python scripts/contar.py --execucao cenario/reduzido/sem_render --cenario cenario/reduzido --detector oraculo \
      --rastreadores chao --p_det 0.8 --sigma_px 1.0 --fp 0.3 --saida resultados/contagem_oraculo.json
Gera <saida>.json, <saida>_deteccoes.csv (deteccoes de trilhas associadas a faixas e clusters, rastreador
principal), <saida>_cache_det.npz (deteccoes + GMC) e, com --figura, um mapa PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SIM = AQUI.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent))

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from cerise_sim.avaliacao import ap_quadros, avaliar_contagem, pr_quadros, rotular_faixa  # noqa: E402
from cerise_sim.botsort_chao import (ParamsBotSort, carregar_cache, contar_execucao_botsort,  # noqa: E402
                                     detectar_e_gmc, gmc_por_pose, salvar_cache)
from cerise_sim.contagem import DetectorOraculo, DetectorYOLO, ParamsContagem, contar_execucao  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402

CRITERIOS = {"iou0.5": dict(iou_min=0.5), "iou0.3": dict(iou_min=0.3), "iou0.3_ou_centro6px": dict(iou_min=0.3, dist_centro_px=6.0)}


def poli(lista):
    ps = [Polygon(c) for c in lista]
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


def pr_total(logs, gts, ids):
    pr = {}
    for nome, kw in CRITERIOS.items():
        tot = {"tp": 0, "fp": 0, "fn": 0}
        por_cam = {}
        errs = []
        for cam in ids:
            r = pr_quadros(logs[cam], gts[cam], **kw)
            por_cam[cam] = r
            for k in tot:
                tot[k] += r[k]
            if r["mae_contagem_quadro"] is not None:
                errs.append((r["quadros"], r["mae_contagem_quadro"], r["vies_contagem_quadro"]))
        P = tot["tp"] / max(tot["tp"] + tot["fp"], 1); R = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
        nq = sum(e[0] for e in errs)
        pr[nome] = {**tot, "precisao": P, "revocacao": R, "f1": 2 * P * R / max(P + R, 1e-9),
                    "mae_contagem_quadro": sum(e[0] * e[1] for e in errs) / max(nq, 1),
                    "vies_contagem_quadro": sum(e[0] * e[2] for e in errs) / max(nq, 1), "por_camera": por_cam}
    return pr


def analisar_clusters(res, gts, cen, regioes, p, plano_tipo):
    """Duplicatas (entre drones x mesmo drone), fusoes entre drones, efeito da fronteira e serie temporal."""
    faixas = res["faixas"]
    rot = {(c, f.id): rotular_faixa(f, gts.get(c, {}))[0] for c, fs in faixas.items() for f in fs}
    idx = {(c, f.id): f for c, fs in faixas.items() for f in fs}
    front = shapely.union_all([g.boundary for g in regioes])
    banda = front.buffer(p.banda_m)
    cls = []
    for cl in res["agregacao"].get("clusters", []):
        membros = [tuple(m) for m in cl]
        v = Counter(rot.get(m) for m in membros)
        lab = v.most_common(1)[0][0] if v else None
        drones = sorted({m[0] for m in membros})
        P = np.mean([idx[m].posicao_representativa() for m in membros], axis=0)
        t0 = min(idx[m].t_ini for m in membros)
        cls.append({"rotulo": lab, "drones": drones, "t_ini": float(t0), "xy": P.tolist(),
                    "na_banda": bool(banda.contains(shapely.Point(P)))})
    por_rot = {}
    for k, c in enumerate(cls):
        if c["rotulo"] is not None:
            por_rot.setdefault(c["rotulo"], []).append(k)
    dup_entre, dup_mesmo, dup_banda = 0, 0, 0
    dup_tipos = Counter()
    for lab, ks in por_rot.items():
        if len(ks) < 2:
            continue
        ks = sorted(ks, key=lambda k: cls[k]["t_ini"])
        for k in ks[1:]:
            entre = not (set(cls[k]["drones"]) & set().union(*[set(cls[j]["drones"]) for j in ks if j != k]))
            dup_entre += int(entre); dup_mesmo += int(not entre)
            dup_banda += int(cls[k]["na_banda"])
            dup_tipos[plano_tipo.get(lab, "?")] += 1
    multi = sum(1 for c in cls if len(c["drones"]) > 1)
    # fronteira: pessoas do tipo "cruza_fronteira" vistas e quantos clusters receberam
    cruz = [pid for pid, t in plano_tipo.items() if t == "cruza_fronteira"]
    clusters_por_pessoa_cruz = {pid: len(por_rot.get(pid, [])) for pid in cruz}
    return {
        "clusters": len(cls), "clusters_com_mais_de_um_drone": multi, "clusters_na_banda": sum(c["na_banda"] for c in cls),
        "duplicatas_entre_drones": dup_entre, "duplicatas_mesmo_drone": dup_mesmo, "duplicatas_na_banda": dup_banda,
        "duplicatas_por_tipo_de_pessoa": dict(dup_tipos), "clusters_por_pessoa_cruza_fronteira": clusters_por_pessoa_cruz,
    }, cls


def serie_temporal(ex, gts, ids, logs, cls, passo_s=1.0):
    """Contagem acumulada estimada (clusters pela 1a aparicao; a agregacao e offline) x pessoas distintas ja
    vistas (gabarito) no tempo, e contagem atual por drone (trilhas no quadro x caixas de gabarito)."""
    primeira = {}
    por_drone = {}
    for cam in ids:
        t_q = {q.frame: q.t for q in ex.quadros(cam)}
        tt, ne, ng = [], [], []
        for f in sorted(logs[cam]):
            t = t_q[f]
            ids_g = gts[cam].get(f, ([], None, None))[0]
            for pid in ids_g:
                primeira[pid] = min(primeira.get(pid, 1e9), t)
            tt.append(round(t, 3)); ne.append(int(len(logs[cam][f][0]))); ng.append(len(ids_g))
        por_drone[cam] = {"t": tt, "estimada": ne, "gabarito": ng}
    tmax = max([max(v["t"]) for v in por_drone.values() if v["t"]] or [0])
    T = np.arange(0, tmax + passo_s, passo_s)
    t_cl = np.sort([c["t_ini"] for c in cls]) if cls else np.zeros(0)
    t_pg = np.sort(list(primeira.values())) if primeira else np.zeros(0)
    return {"t": T.round(3).tolist(),
            "estimada_acumulada": np.searchsorted(t_cl, T, side="right").tolist(),
            "vistas_acumulada_gabarito": np.searchsorted(t_pg, T, side="right").tolist(),
            "por_drone": por_drone}


def avaliar_deteccoes_brutas(dets, gts, ids, quadros_usados, taus):
    out = {"ap": {}, "por_limiar": {}}
    for nome, kw in CRITERIOS.items():
        aps = [ap_quadros(dets[cam], gts[cam], sorted(quadros_usados[cam]), **kw) for cam in ids]
        # AP combinado: concatena todas as cameras
        comb_det, comb_gt, fr_all = {}, {}, []
        for cam in ids:
            for f in quadros_usados[cam]:
                comb_det[(cam, f)] = dets[cam].get(f, (np.zeros((0, 4)), np.zeros(0)))
                comb_gt[(cam, f)] = gts[cam].get(f, ([], np.zeros((0, 4)), np.zeros(0)))
                fr_all.append((cam, f))
        regs = ap_quadros(comb_det, comb_gt, fr_all, **kw)
        regs["por_camera"] = {cam: a["ap"] for cam, a in zip(ids, aps)}
        out["ap"][nome] = regs
    for tau in taus:
        logs = {cam: {f: (dets[cam][f][0][dets[cam][f][1] >= tau], dets[cam][f][1][dets[cam][f][1] >= tau], None)
                      for f in quadros_usados[cam] if f in dets[cam]} for cam in ids}
        out["por_limiar"][f"{tau:.3f}"] = pr_total(logs, gts, ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--detector", choices=["yolo", "oraculo"], default="yolo")
    ap.add_argument("--rastreadores", default="botsort,chao",
                    help="lista; o 1o e o principal. botsort (GMC das imagens), botsort_pose (GMC da telemetria), "
                         "botsort_semgmc (sem GMC), chao (rastreador proprio no solo)")
    ap.add_argument("--pesos", default="resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf_cache", type=float, default=0.01, help="conf minima das deteccoes em cache (como no val)")
    ap.add_argument("--iou_nms", type=float, default=0.6)
    ap.add_argument("--conf_chao", type=float, default=0.175, help="conf minima para o rastreador no chao")
    ap.add_argument("--bs_cal", default="0.15,0.05,0",
                    help="modo botsort_cal: 'high=new,low,fuse_score' (calibrado no voo a 40 m pela varredura a posteriori)")
    ap.add_argument("--taus", default="0.175,0.35,0.40", help="limiares para P/R/MAE por quadro das deteccoes brutas")
    ap.add_argument("--cache_det", default=None, help="npz de deteccoes+GMC (padrao <saida>_cache_det.npz; reaproveitado se existir)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--p_det", type=float, default=1.0)
    ap.add_argument("--sigma_px", type=float, default=0.0)
    ap.add_argument("--fp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figura", action="store_true")
    for f in fields(ParamsContagem):
        ap.add_argument("--" + f.name, type=type(f.default) if not isinstance(f.default, bool) else (lambda s: s.lower() in ("1", "true", "sim")),
                        default=f.default)
    for f in fields(ParamsBotSort):
        if f.name in ("gmc_method",):
            continue
        ap.add_argument("--bs_" + f.name, type=type(f.default) if not isinstance(f.default, bool) else (lambda s: s.lower() in ("1", "true", "sim")),
                        default=f.default)
    a = ap.parse_args()
    p = ParamsContagem(**{f.name: getattr(a, f.name) for f in fields(ParamsContagem)})
    pb = ParamsBotSort(**{f.name: getattr(a, "bs_" + f.name) for f in fields(ParamsBotSort) if f.name != "gmc_method"})
    cen = json.loads((Path(a.cenario) / "cenario.json").read_text(encoding="utf-8"))
    plano = json.loads((Path(a.cenario) / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
    plano_tipo = {q["id"]: q["tipo"] for q in plano}
    area = poli(cen["area_enu"])
    ex = Execucao(a.execucao)
    ids = [d for d in cen["regioes_enu"] if d in ex.cameras("drone")]
    regioes = [poli(cen["regioes_enu"][d]) for d in ids]
    gts = {c: ex.caixas_gt(c) for c in ids}
    out = Path(a.saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_ini = time.time()

    # ------------------------------------------------------------------ deteccoes (cache) + GMC
    tempos_det = None
    cache = Path(a.cache_det) if a.cache_det else out.with_name(out.stem + "_cache_det.npz")
    if a.detector == "yolo":
        sidecar = cache.with_suffix(".tempo.json")
        if cache.exists():
            dets, gmcs = carregar_cache(cache)
            print(f"cache de deteccoes reaproveitado: {cache}", flush=True)
            if sidecar.exists():
                tempos_det = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            dev = int(a.device) if a.device.isdigit() else a.device
            yolo = DetectorYOLO(a.pesos, a.imgsz, a.conf_cache, iou=a.iou_nms, device=dev)
            yolo(np.zeros((64, 64, 3), np.uint8))  # aquecimento (nao entra no tempo)
            dets, gmcs, tempos_det = detectar_e_gmc(ex, ids, yolo, gmc=True, log=lambda s: print(s, flush=True))
            salvar_cache(cache, dets, gmcs)
            sidecar.write_text(json.dumps(tempos_det), encoding="utf-8")
    else:
        dets, gmcs = {}, {}
        for cam in ids:
            K = ex.intrinsecos(cam)
            o = DetectorOraculo(gts[cam], K.largura, K.altura, a.p_det, a.sigma_px, a.fp, seed=a.seed + sum(map(ord, cam)))
            dets[cam] = {q.frame: o(q.frame) for q in ex.quadros(cam)}
            gmcs[cam] = {f: np.eye(2, 3) for f in dets[cam]}

    quadros_usados = {cam: [q.frame for q in ex.quadros(cam) if (q.cobre or not p.somente_cobertura)] for cam in ids}
    saida = {
        "execucao": str(a.execucao), "cenario": cen["nome"], "simulador": ex.meta.get("simulador"),
        "modo_pessoas": ex.meta.get("modo_pessoas"), "detector": a.detector,
        "detector_params": ({"pesos": a.pesos, "imgsz": a.imgsz, "conf_cache": a.conf_cache, "iou_nms": a.iou_nms, "fp32": True}
                            if a.detector == "yolo" else {"p_det": a.p_det, "sigma_px": a.sigma_px, "fp_por_quadro": a.fp}),
        "parametros_contagem": asdict(p), "parametros_botsort": asdict(pb), "drones": ids, "fps": ex.fps,
        "quadros_usados_por_drone": {c: len(v) for c, v in quadros_usados.items()},
        "quadros_totais_por_drone": {c: len(ex.quadros(c)) for c in ids},
        "s_render_por_quadro": ex.meta.get("s_render_por_quadro"), "s_por_quadro_sim": ex.meta.get("s_por_quadro"),
    }
    if tempos_det:
        n = max(tempos_det["n"], 1)
        saida["tempo_deteccao"] = {"quadros": tempos_det["n"], "ms_leitura": 1000 * tempos_det["leitura_s"] / n,
                                   "ms_deteccao": 1000 * tempos_det["det_s"] / n, "ms_gmc": 1000 * tempos_det["gmc_s"] / n,
                                   "fps_deteccao": n / max(tempos_det["det_s"], 1e-9),
                                   "fps_leitura_det_gmc": n / max(tempos_det["leitura_s"] + tempos_det["det_s"] + tempos_det["gmc_s"], 1e-9)}
    if a.detector == "yolo":
        taus = [float(x) for x in a.taus.split(",") if x]
        saida["deteccao_bruta"] = avaliar_deteccoes_brutas(dets, gts, ids, quadros_usados, taus)

    # ------------------------------------------------------------------ rastreio + contagem unica
    modos = [m for m in a.rastreadores.split(",") if m]
    resultados = {}
    for modo in modos:
        t0 = time.time()
        if modo.startswith("botsort"):
            if modo == "botsort_pose" or (modo == "botsort" and a.detector == "oraculo"):
                g = {cam: gmc_por_pose(ex, cam, p.altura_ref_m) for cam in ids}
            elif modo == "botsort_semgmc":
                g = {cam: {f: np.eye(2, 3) for f in dets[cam]} for cam in ids}
            else:
                g = gmcs
            pbm = pb
            if modo == "botsort_cal":
                h, lo, fu = a.bs_cal.split(",")
                pbm = replace(pb, track_high_thresh=float(h), new_track_thresh=float(h), track_low_thresh=float(lo),
                              fuse_score=fu.strip().lower() in ("1", "true", "sim"))
            res = contar_execucao_botsort(ex, dets, g, area, regioes, ids, p, pbm)
            saida.setdefault("parametros_botsort_por_modo", {})[modo] = asdict(pbm)
        else:
            def fabrica(cam, _d=dets):
                def det(q):
                    c, s = _d[cam].get(q.frame, (np.zeros((0, 4)), np.zeros(0)))
                    m = s >= (a.conf_chao if a.detector == "yolo" else 0)
                    return c[m], s[m]
                return det
            res = contar_execucao(ex, fabrica, area, regioes, ids, p)
        t_rast = time.time() - t0
        av = avaliar_contagem(res, gts, len(ex.pessoas()))
        pr = pr_total(res["logs"], gts, ids)
        an, cls = analisar_clusters(res, gts, cen, regioes, p, plano_tipo)
        ag = res["agregacao"]
        resultados[modo] = (res, av, cls)
        saida[modo] = {
            "avaliacao_contagem": av, "deteccao_por_quadro_trilhas": pr, "analise_clusters": an,
            "agregacao": {k: v for k, v in ag.items() if k != "clusters"},
            "faixas_por_drone": {c: len(fs) for c, fs in res["faixas"].items()},
            "tempo_rastreio_agregacao_s": t_rast,
            "serie_temporal": serie_temporal(ex, gts, ids, res["logs"], cls),
            "clusters": cls,
        }
        print(f"[{modo}] contagem unica {av['contagem_proposta']} x vistas {av['pessoas_vistas_gabarito']} "
              f"(cena {av['pessoas_na_cena']}); dup {av['clusters_duplicados']} (entre drones {an['duplicatas_entre_drones']}), "
              f"falsos {av['clusters_falsos']}, perdidas {av['pessoas_perdidas']}; P/R(iou0.3|6px) "
              f"{pr['iou0.3_ou_centro6px']['precisao']:.3f}/{pr['iou0.3_ou_centro6px']['revocacao']:.3f}; "
              f"MAE/quadro {pr['iou0.3_ou_centro6px']['mae_contagem_quadro']:.2f}", flush=True)
    saida["principal"] = modos[0]
    saida["tempo_total_s"] = time.time() - t_ini
    out.write_text(json.dumps(saida, indent=1, default=float), encoding="utf-8")

    # deteccoes associadas a faixas e clusters: <stem>_deteccoes.csv (principal) e <stem>_deteccoes_<modo>.csv
    for k_modo, modo in enumerate(modos):
        res, av, cls = resultados[modo]
        cl_de = {}
        for k, cl in enumerate(res["agregacao"].get("clusters", [])):
            for m in cl:
                cl_de[tuple(m)] = k
        nome = out.stem + ("_deteccoes.csv" if k_modo == 0 else f"_deteccoes_{modo}.csv")
        with open(out.with_name(nome), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["camera", "frame", "t", "faixa", "cluster", "x1", "y1", "x2", "y2", "X", "Y", "score"])
            for cam, fs in res["faixas"].items():
                for fx in fs:
                    for fr, t, b, xy, s in zip(fx.frames, fx.t, fx.caixas, fx.xy, fx.scores):
                        w.writerow([cam, fr, round(t, 3), fx.id, cl_de.get((cam, fx.id), -1), *np.round(b, 1), *np.round(xy, 2), round(s, 3)])
    res, av, cls = resultados[modos[0]]
    resumo = {m: {k: saida[m]["avaliacao_contagem"][k] for k in ("contagem_proposta", "pessoas_vistas_gabarito", "pessoas_na_cena",
                                                                  "erro_pct_vs_vistas", "clusters_duplicados", "clusters_falsos", "pessoas_perdidas")}
              for m in modos}
    print(json.dumps({"resumo": resumo, "tempo_total_s": saida["tempo_total_s"], "tempo_deteccao": saida.get("tempo_deteccao"),
                      "ap": {k: v["ap"] for k, v in saida.get("deteccao_bruta", {}).get("ap", {}).items()}}, indent=1, default=float))
    if a.figura:
        figura(cen, ex, res, av, out.with_suffix(".png"), a, modos[0])


def figura(cen, ex, res, av, caminho, a, modo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    cores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for i, (d, polys) in enumerate(cen["regioes_enu"].items()):
        for pl in polys:
            ax.add_patch(MP(pl, closed=True, fc=cores[i % 4], alpha=0.12, ec=cores[i % 4], lw=1))
    gab = ex.gabarito()
    ult = gab[max(gab)]
    xy = np.array(list(ult.values()))
    ax.plot(xy[:, 0], xy[:, 1], "k.", ms=4, label=f"pessoas (gabarito, posição final) N={len(ult)}")
    gts = {c: ex.caixas_gt(c) for c in res["faixas"]}
    rot = {(c, f.id): rotular_faixa(f, gts[c])[0] for c, fs in res["faixas"].items() for f in fs}
    vistos = set()
    marc = {"unico": ("o", "#2ca02c"), "dup": ("s", "#ff7f0e"), "falso": ("x", "#d62728")}
    usados = set()
    for cl in res["agregacao"].get("clusters", []):
        labs = [rot.get(tuple(m)) for m in cl]
        lab = max(set(labs), key=labs.count)
        fx = [f for c, fs in res["faixas"].items() for f in fs if (c, f.id) in {tuple(m) for m in cl}]
        P = np.mean([f.posicao_representativa() for f in fx], axis=0)
        tipo = "falso" if lab is None else ("dup" if lab in vistos else "unico")
        vistos.add(lab)
        mk, c = marc[tipo]
        ax.plot(P[0], P[1], mk, mfc="none" if tipo == "unico" else c, mec=c, ms=7,
                label={"unico": "cluster correto", "dup": "duplicata", "falso": "falso positivo"}[tipo] if tipo not in usados else None)
        usados.add(tipo)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("x leste (m)"); ax.set_ylabel("y norte (m)")
    ax.set_title(f"Contagem única global ({a.detector}, {modo}): estimada {av['contagem_proposta']} x vistas {av['pessoas_vistas_gabarito']} "
                 f"(cena {av['pessoas_na_cena']}); dup {av['clusters_duplicados']}, falsos {av['clusters_falsos']}, perdidas {av['pessoas_perdidas']}",
                 fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
