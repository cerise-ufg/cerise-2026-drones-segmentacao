"""Etapa 3 (CPU): rastreamento multiobjeto offline sobre as deteccoes em cache, com os rastreadores do proprio
ultralytics (BYTETracker = ByteTrack, BOTSORT = BoT-SORT com GMC sparseOptFlow, sem ReID), e avaliacao.

Protocolo sem vazamento:
  1. grade de parametros no VAL (7 sequencias): track_high_thresh x track_low_thresh x track_buffer, com
     new_track_thresh = track_high_thresh e os demais parametros nos valores padrao do ultralytics;
     escolhe-se, para cada (detector, rastreador), a configuracao de maior HOTA combinado no val;
  2. no VAL tambem se escolhem: o limiar de confianca da contagem por quadro a partir das deteccoes (menor MAE)
     e o comprimento minimo de trilha L para a contagem de pessoas unicas (menor erro relativo medio);
  3. tudo e aplicado sem alteracao ao TEST-DEV (17 sequencias), junto com a configuracao padrao do ultralytics
     como referencia.
Gabarito: pedestrian(1)+people(2), score=1; caixas (gabarito e rastreador) com >50% da area em regioes
ignoradas (classes 0 e 11) sao descartadas, como no protocolo do VisDrone.
Uso: python 30_rastrear.py   (NPROC processos)
"""
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

import metricas_mot as mm

BASE = "/w/validacao_videos"
MOT = "/w/mot"
RES = f"{BASE}/resultados"
VARIANTES = ["ft960", "ft1280", "coco960", "coco1280"]
RASTREADORES = ["bytetrack", "botsort"]
PADRAO = {"track_high_thresh": 0.25, "track_low_thresh": 0.1, "new_track_thresh": 0.25, "track_buffer": 30}
GRADE = [{"track_high_thresh": h, "track_low_thresh": l, "new_track_thresh": h, "track_buffer": b}
         for h in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7) for l in (0.05, 0.1) if l < h for b in (30, 60)]
LS = (1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180)  # comprimentos minimos de trilha (quadros) para a contagem unica
TAUS = np.round(np.arange(0.05, 0.701, 0.025), 3)  # limiares da contagem por quadro a partir das deteccoes
CFG_BASE = {n: YAML.load(check_yaml(f"{n}.yaml")) for n in RASTREADORES}
SEQS = {s: sorted(os.path.basename(d) for d in glob.glob(f"{MOT}/VisDrone2019-MOT-{s}/sequences/*")) for s in ("val", "test-dev")}
_cache = {}


def dados(split, seq):
    k = (split, seq)
    if k not in _cache:
        T = len(os.listdir(f"{MOT}/VisDrone2019-MOT-{split}/sequences/{seq}"))
        gt, ign = mm.ler_gabarito(f"{MOT}/VisDrone2019-MOT-{split}/annotations/{seq}.txt", T)
        gtf = {}
        for t in range(1, T + 1):
            gi, gb, _ = gt[t]
            gtf[t] = mm.filtrar_ignorado(gi, gb, ign[t])
        dets = {}
        for v in VARIANTES:
            z = np.load(f"{BASE}/cache_det/{v}/{split}/{seq}.npz")
            dets[v] = (z["quadro"], z["xyxy"], z["conf"], int(z["altura"]), int(z["largura"]))
        W = np.load(f"{BASE}/cache_gmc/{split}/{seq}.npy")
        _cache[k] = {"T": T, "gt": gtf, "ign": ign, "dets": dets, "gmc": W}
    return _cache[k]


def rastrear(d, variante, nome, cfg):
    q, xy, cf, H, Wd = d["dets"][variante]
    args = IterableSimpleNamespace(**{**CFG_BASE[nome], **cfg})
    trk = BOTSORT(args) if nome == "botsort" else BYTETracker(args)
    estado = {"t": 0}
    if nome == "botsort":  # GMC pre-calculada (20_gmc_condicoes.py), identica para todas as configuracoes
        trk.gmc.apply = lambda img, dets=None: d["gmc"][estado["t"]]
    img = np.zeros((1, 1, 3), np.uint8)
    lim = np.searchsorted(q, np.arange(1, d["T"] + 2))
    out = {}
    for t in range(1, d["T"] + 1):
        a, b = lim[t - 1], lim[t]
        estado["t"] = t - 1
        r = trk.update(Boxes(np.c_[xy[a:b], cf[a:b], np.zeros(b - a, np.float32)], (H, Wd)), img)
        if len(r):
            ids, bx, sc = r[:, 4].astype(int), r[:, :4].astype(float), r[:, 5]
            ids, bx, sc = mm.filtrar_ignorado(ids, bx, d["ign"][t], extra=sc)
        else:
            ids, bx, sc = np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)
        out[t] = (ids, bx, sc)
    return out


def contar_unicos(out, gt, T, L):
    """Pessoas unicas: ids do rastreador com >= L quadros (ja sem regioes ignoradas) e ids do gabarito."""
    comp = {}
    for t in range(1, T + 1):
        for i in out[t][0].tolist():
            comp[i] = comp.get(i, 0) + 1
    gt_ids = set()
    for t in range(1, T + 1):
        gt_ids.update(gt[t][0].tolist())
    return sum(1 for c in comp.values() if c >= L), len(gt_ids), comp


def tarefa(arg):
    variante, nome, ci, cfg, split, seq, guardar = arg
    d = dados(split, seq)
    t0 = time.time()
    out = rastrear(d, variante, nome, cfg)
    t_trk = time.time() - t0
    T = d["T"]
    s = mm.montar_sequencia(d["gt"], {t: out[t][:2] for t in out}, T)
    pares = set()
    bruto = mm.avaliar(s, pares=pares)
    _, n_gt, comp = contar_unicos(out, d["gt"], T, 1)
    unicos = {L: sum(1 for c in comp.values() if c >= L) for L in LS}
    cont_trk = np.array([len(out[t][0]) for t in range(1, T + 1)])
    cont_gt = np.array([len(d["gt"][t][0]) for t in range(1, T + 1)])
    # decomposicao da contagem unica (L=1): ids falsos, excesso por troca/fragmentacao, pessoas nunca rastreadas
    gt_casados = {g for g, _ in pares}; trk_casados = {k for _, k in pares}
    decomp = {"trk_ids": s["n_trk_ids"], "trk_ids_casados": len(trk_casados), "trk_ids_falsos": s["n_trk_ids"] - len(trk_casados),
              "gt_ids": s["n_gt_ids"], "gt_ids_rastreados": len(gt_casados), "gt_ids_perdidos": s["n_gt_ids"] - len(gt_casados),
              "excesso_troca_fragmentacao": len(trk_casados) - len(gt_casados),
              "ids_trk_por_pessoa_rastreada": len(pares) / max(1, len(gt_casados))}
    r = {"variante": variante, "rastreador": nome, "cfg_idx": ci, "cfg": cfg, "split": split, "seq": seq, "T": T,
         "bruto": bruto, "unicos_pred": unicos, "unicos_gt": n_gt, "decomp_L1": decomp,
         "cont_trk": cont_trk, "cont_gt": cont_gt, "tempo_rastreio_s": t_trk}
    if guardar:
        r["saida"] = {t: (out[t][0], out[t][1].astype(np.float32), out[t][2].astype(np.float32)) for t in out}
        r["comp"] = comp
    return r


def contagem_deteccoes(split, seq, variante):
    """Contagem por quadro a partir das deteccoes (sem rastreamento) para cada limiar TAU."""
    d = dados(split, seq)
    q, xy, cf, _, _ = d["dets"][variante]
    ign_mask = np.zeros(len(q), bool)
    for t in range(1, d["T"] + 1):
        m = q == t
        if m.any() and len(d["ign"][t]):
            ign_mask[m] = mm.fracao_em_ignorado(xy[m], d["ign"][t]) > 0.5
    pred = np.zeros((len(TAUS), d["T"]), int)
    for i, tau in enumerate(TAUS):
        ok = (cf >= tau) & ~ign_mask
        pred[i] = np.bincount(q[ok] - 1, minlength=d["T"])[:d["T"]]
    gt = np.array([len(d["gt"][t][0]) for t in range(1, d["T"] + 1)])
    return pred, gt


def resumo_lista(rs):
    m = mm.resumir([r["bruto"] for r in rs])
    m["contagem_quadro_trk"] = mm.erros_contagem(np.concatenate([r["cont_trk"] for r in rs]), np.concatenate([r["cont_gt"] for r in rs]))
    return m


def erro_unico(rs, L):
    e = [(r["unicos_pred"][L] - r["unicos_gt"]) / r["unicos_gt"] for r in rs if r["unicos_gt"] > 0]
    return float(np.mean(np.abs(e))), float(np.mean(e))


def sem_arrays(r):
    x = {k: v for k, v in r.items() if k not in ("bruto", "cont_trk", "cont_gt", "saida", "comp")}
    x.update(mm.resumir([r["bruto"]]))
    x["contagem_quadro_trk"] = mm.erros_contagem(r["cont_trk"], r["cont_gt"])
    return x


if __name__ == "__main__":
    NP = int(os.environ.get("NPROC", "4"))
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    sel = {"grade": GRADE, "padrao_ultralytics": PADRAO, "base_yaml": CFG_BASE, "Ls": LS}

    # ---------------- (a) contagem por quadro a partir das deteccoes: limiar escolhido no val
    cont_det = {}
    for v in VARIANTES:
        pv = [contagem_deteccoes("val", s, v) for s in SEQS["val"]]
        P = np.concatenate([p for p, _ in pv], 1); G = np.concatenate([g for _, g in pv])
        maes = [float(np.abs(P[i] - G).mean()) for i in range(len(TAUS))]
        ib = int(np.argmin(maes)); tau = float(TAUS[ib])
        cont_det[v] = {"tau_val": tau, "mae_val_por_tau": dict(zip(map(float, TAUS), maes)), "val": {}, "test-dev": {}}
        for split in ("val", "test-dev"):
            Ps, Gs = [], []
            for s in SEQS[split]:
                p, g = contagem_deteccoes(split, s, v)
                cont_det[v][split][s] = mm.erros_contagem(p[ib], g)
                Ps.append(p[ib]); Gs.append(g)
            cont_det[v][split]["_agregado"] = mm.erros_contagem(np.concatenate(Ps), np.concatenate(Gs))
        print(v, "tau", tau, "MAE val", round(maes[ib], 3), "MAE test", round(cont_det[v]["test-dev"]["_agregado"]["MAE"], 3), flush=True)
    json.dump(cont_det, open(f"{RES}/contagem_deteccoes.json", "w"), indent=1)

    # ---------------- (b) grade no val
    tarefas = [(v, n, ci, cfg, "val", s, False) for v in VARIANTES for n in RASTREADORES
               for ci, cfg in enumerate(GRADE) for s in SEQS["val"]]
    print("tarefas da grade:", len(tarefas), flush=True)
    res = {}
    with Pool(NP) as p:
        for i, r in enumerate(p.imap_unordered(tarefa, tarefas, chunksize=2)):
            res.setdefault((r["variante"], r["rastreador"], r["cfg_idx"]), []).append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(tarefas)} {time.time()-t0:.0f}s", flush=True)
    linhas = []
    for (v, n, ci), rs in sorted(res.items()):
        m = resumo_lista(rs)
        eu = {L: erro_unico(rs, L) for L in LS}
        linhas.append({"variante": v, "rastreador": n, "cfg_idx": ci, **GRADE[ci],
                       **{k: m[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "MOTP")},
                       "MAE_quadro_trk": m["contagem_quadro_trk"]["MAE"],
                       **{f"erro_rel_unico_abs_L{L}": eu[L][0] for L in LS}})
    import csv
    with open(f"{RES}/grade_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys())); w.writeheader(); w.writerows(linhas)
    escolha = {}
    for v in VARIANTES:
        for n in RASTREADORES:
            ls = [x for x in linhas if x["variante"] == v and x["rastreador"] == n]
            b = max(ls, key=lambda x: x["HOTA"])
            Lb = min(LS, key=lambda L: b[f"erro_rel_unico_abs_L{L}"])
            escolha[f"{v}|{n}"] = {"cfg_idx": b["cfg_idx"], "cfg": GRADE[b["cfg_idx"]], "HOTA_val": b["HOTA"], "L": Lb,
                                   "erro_rel_unico_abs_val": b[f"erro_rel_unico_abs_L{Lb}"]}
            print("escolha", v, n, escolha[f"{v}|{n}"], flush=True)
    sel["escolha"] = escolha
    sel["tempo_grade_s"] = time.time() - t0
    json.dump(sel, open(f"{RES}/selecao_val.json", "w"), indent=1)

    # ---------------- (c) configuracao escolhida e padrao, val e test-dev (com saidas guardadas)
    final = []
    for v in VARIANTES:
        for n in RASTREADORES:
            e = escolha[f"{v}|{n}"]
            for rotulo, cfg in (("escolhida_val", e["cfg"]), ("padrao", PADRAO)):
                for split in ("val", "test-dev"):
                    for s in SEQS[split]:
                        final.append((v, n, rotulo, cfg, split, s, True))
    agreg, por_seq = {}, []
    os.makedirs(f"{BASE}/rastreios", exist_ok=True)
    with Pool(NP) as p:
        for r in p.imap_unordered(tarefa, final, chunksize=1):
            rot = r["cfg_idx"]
            L = escolha[f"{r['variante']}|{r['rastreador']}"]["L"]
            agreg.setdefault((r["variante"], r["rastreador"], rot, r["split"]), []).append(r)
            x = sem_arrays(r); x["rotulo"] = rot; x["L_escolhido"] = L
            x["unicos_pred_L"] = r["unicos_pred"][L]
            por_seq.append(x)
            if rot == "escolhida_val":  # guarda a trilha em formato MOTChallenge (para figuras/videos)
                pasta = f"{BASE}/rastreios/{r['variante']}_{r['rastreador']}/{r['split']}"
                os.makedirs(pasta, exist_ok=True)
                with open(f"{pasta}/{r['seq']}.txt", "w") as f:
                    for t, (ids, bx, sc) in sorted(r["saida"].items()):
                        for i, b, c in zip(ids, bx, sc):
                            f.write(f"{t},{i},{b[0]:.1f},{b[1]:.1f},{b[2]-b[0]:.1f},{b[3]-b[1]:.1f},{c:.3f},-1,-1,-1\n")
    out = {}
    for (v, n, rot, split), rs in sorted(agreg.items()):
        L = escolha[f"{v}|{n}"]["L"]
        m = resumo_lista(rs)
        tot_gt = sum(r["unicos_gt"] for r in rs)
        dec = {k: sum(r["decomp_L1"][k] for r in rs) for k in ("trk_ids", "trk_ids_casados", "trk_ids_falsos", "gt_ids",
                                                             "gt_ids_rastreados", "gt_ids_perdidos", "excesso_troca_fragmentacao")}
        m["unicos"] = {"gt_total": tot_gt, "pred_total_L1": sum(r["unicos_pred"][1] for r in rs),
                       "pred_total_L": sum(r["unicos_pred"][L] for r in rs), "L": L,
                       "erro_rel_abs_medio_L1": erro_unico(rs, 1)[0], "vies_rel_medio_L1": erro_unico(rs, 1)[1],
                       "erro_rel_abs_medio_L": erro_unico(rs, L)[0], "vies_rel_medio_L": erro_unico(rs, L)[1],
                       "decomposicao_L1": dec}
        m["tempo_rastreio_ms_por_quadro"] = 1000 * sum(r["tempo_rastreio_s"] for r in rs) / sum(r["T"] for r in rs)
        m["n_seq"] = len(rs)
        out[f"{v}|{n}|{rot}|{split}"] = m
        print(f"{v:9s} {n:9s} {rot:13s} {split:8s} HOTA {m['HOTA']:.3f} MOTA {m['MOTA']:.3f} IDF1 {m['IDF1']:.3f} "
              f"IDSW {m['IDSW']} MAEq {m['contagem_quadro_trk']['MAE']:.2f} unicos gt {tot_gt} pred L1 {m['unicos']['pred_total_L1']} "
              f"L{L} {m['unicos']['pred_total_L']}", flush=True)
    json.dump(out, open(f"{RES}/metricas_agregadas.json", "w"), indent=1, default=float)
    json.dump(por_seq, open(f"{RES}/metricas_por_sequencia.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print("tempo total", round(time.time() - t0), "s", flush=True)
    print("FIM_RASTREIO", flush=True)
