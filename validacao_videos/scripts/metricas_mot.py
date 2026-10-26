"""Metricas de rastreamento multiobjeto (HOTA, CLEAR/MOTA, Identity/IDF1) e utilitarios do VisDrone2019-MOT.

As tres metricas seguem, linha a linha, as definicoes e a implementacao de referencia do TrackEval
(Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking", IJCV 2021;
github.com/JonathonLuiten/TrackEval, arquivos trackeval/metrics/{hota,clear,identity}.py), reescritas aqui
porque o TrackEval original nao e compativel com numpy >= 1.24 sem modificacoes. MOTA e IDF1 sao conferidos
contra o py-motmetrics (PyPI) em 31_conferir_motmetrics.py.

Formato de uma sequencia (dict):
  T: numero de quadros; gt_ids[t], trk_ids[t]: ids inteiros contiguos 0..n-1; sim[t]: IoU (n_gt_t x n_trk_t)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = np.finfo("float").eps
ALPHAS = np.arange(0.05, 0.99, 0.05)  # 19 limiares de localizacao (como no TrackEval)
CLASSES_PESSOA = (1, 2)  # VisDrone: 1 pedestrian, 2 people
CLASSES_IGNORAR = (0, 11)  # 0 ignored regions, 11 others


# ----------------------------------------------------------------------------------------------- utilitarios
def iou(a, b):
    """IoU entre caixas xyxy (n x 4) e (m x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    uni = aa[:, None] + ab[None, :] - inter
    return np.where(uni > 0, inter / np.maximum(uni, 1e-12), 0.0)


def fracao_em_ignorado(xyxy, ign):
    """Maior fracao da area de cada caixa contida numa regiao ignorada (xyxy)."""
    if len(xyxy) == 0 or len(ign) == 0:
        return np.zeros(len(xyxy))
    a = np.asarray(xyxy, float); g = np.asarray(ign, float)
    ix1 = np.maximum(a[:, None, 0], g[None, :, 0]); iy1 = np.maximum(a[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], g[None, :, 2]); iy2 = np.minimum(a[:, None, 3], g[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)
    return (inter / area[:, None]).max(1)


def ler_gabarito(arq, n_quadros):
    """Le a anotacao VisDrone-MOT. Retorna por quadro (1..T): pessoas {ids, xyxy, oclusao} e regioes ignoradas."""
    d = np.loadtxt(arq, delimiter=",", ndmin=2, dtype=np.int64)
    gt, ign = {}, {}
    for t in range(1, n_quadros + 1):
        gt[t] = (np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0, np.int64))
        ign[t] = np.zeros((0, 4))
    if len(d) == 0:
        return gt, ign
    xyxy = np.c_[d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5]].astype(float)
    for t in np.unique(d[:, 0]):
        m = d[:, 0] == t
        p = m & np.isin(d[:, 7], CLASSES_PESSOA) & (d[:, 6] == 1)
        g = m & np.isin(d[:, 7], CLASSES_IGNORAR)
        gt[int(t)] = (d[p, 1], xyxy[p], d[p, 9])
        ign[int(t)] = xyxy[g]
    return gt, ign


def filtrar_ignorado(ids, xyxy, ign, extra=None, lim=0.5):
    """Remove caixas com mais de `lim` da area dentro de regioes ignoradas (protocolo VisDrone)."""
    keep = fracao_em_ignorado(xyxy, ign) <= lim
    if extra is None:
        return ids[keep], xyxy[keep]
    return ids[keep], xyxy[keep], extra[keep]


def montar_sequencia(gt_por_quadro, trk_por_quadro, T):
    """gt/trk_por_quadro: {t: (ids, xyxy)} -> dicionario no formato das metricas (ids contiguos, IoU)."""
    gmap, tmap = {}, {}
    seq = {"T": T, "gt_ids": [], "trk_ids": [], "sim": []}
    for t in range(1, T + 1):
        gi, gb = gt_por_quadro.get(t, (np.zeros(0, int), np.zeros((0, 4))))
        ti, tb = trk_por_quadro.get(t, (np.zeros(0, int), np.zeros((0, 4))))
        seq["gt_ids"].append(np.array([gmap.setdefault(int(i), len(gmap)) for i in gi], int))
        seq["trk_ids"].append(np.array([tmap.setdefault(int(i), len(tmap)) for i in ti], int))
        seq["sim"].append(iou(gb, tb))
    seq["n_gt_ids"], seq["n_trk_ids"] = len(gmap), len(tmap)
    seq["n_gt_dets"] = int(sum(len(x) for x in seq["gt_ids"]))
    seq["n_trk_dets"] = int(sum(len(x) for x in seq["trk_ids"]))
    return seq


# ----------------------------------------------------------------------------------------------- HOTA
def hota(seq):
    ng, nt = seq["n_gt_ids"], seq["n_trk_ids"]
    A = len(ALPHAS)
    r = {k: np.zeros(A) for k in ("TP", "FN", "FP", "LocA", "AssA", "AssRe", "AssPr")}
    if seq["n_trk_dets"] == 0:
        r["FN"][:] = seq["n_gt_dets"]; r["LocA"][:] = 1.0
        return _hota_final(r)
    if seq["n_gt_dets"] == 0:
        r["FP"][:] = seq["n_trk_dets"]; r["LocA"][:] = 1.0
        return _hota_final(r)
    pot = np.zeros((ng, nt)); gcnt = np.zeros((ng, 1)); tcnt = np.zeros((1, nt))
    for g, k, s in zip(seq["gt_ids"], seq["trk_ids"], seq["sim"]):
        den = s.sum(0)[None, :] + s.sum(1)[:, None] - s
        si = np.zeros_like(s); m = den > 0 + EPS
        si[m] = s[m] / den[m]
        pot[g[:, None], k[None, :]] += si
        gcnt[g] += 1; tcnt[0, k] += 1
    gas = pot / (gcnt + tcnt - pot)
    mcounts = [np.zeros((ng, nt)) for _ in ALPHAS]
    for g, k, s in zip(seq["gt_ids"], seq["trk_ids"], seq["sim"]):
        if len(g) == 0:
            r["FP"] += len(k); continue
        if len(k) == 0:
            r["FN"] += len(g); continue
        score = gas[g[:, None], k[None, :]] * s
        mr, mc = linear_sum_assignment(-score)
        for a, al in enumerate(ALPHAS):
            ok = s[mr, mc] >= al - EPS
            ar, ac = mr[ok], mc[ok]
            n = len(ar)
            r["TP"][a] += n; r["FN"][a] += len(g) - n; r["FP"][a] += len(k) - n
            if n:
                r["LocA"][a] += s[ar, ac].sum()
                mcounts[a][g[ar], k[ac]] += 1
    for a in range(A):
        mcount = mcounts[a]
        ass_a = mcount / np.maximum(1, gcnt + tcnt - mcount)
        r["AssA"][a] = (mcount * ass_a).sum() / max(1, r["TP"][a])
        r["AssRe"][a] = (mcount * (mcount / np.maximum(1, gcnt))).sum() / max(1, r["TP"][a])
        r["AssPr"][a] = (mcount * (mcount / np.maximum(1, tcnt))).sum() / max(1, r["TP"][a])
    r["LocA"] = np.maximum(1e-10, r["LocA"]) / np.maximum(1e-10, r["TP"])
    return _hota_final(r)


def _hota_final(r):
    r["DetRe"] = r["TP"] / np.maximum(1, r["TP"] + r["FN"])
    r["DetPr"] = r["TP"] / np.maximum(1, r["TP"] + r["FP"])
    r["DetA"] = r["TP"] / np.maximum(1, r["TP"] + r["FN"] + r["FP"])
    r["HOTA"] = np.sqrt(r["DetA"] * r["AssA"])
    return r


def hota_combinar(lst):
    r = {k: sum(x[k] for x in lst) for k in ("TP", "FN", "FP")}
    for k in ("AssA", "AssRe", "AssPr"):
        r[k] = sum(x[k] * x["TP"] for x in lst) / np.maximum(1e-10, r["TP"])
    r["LocA"] = np.maximum(1e-10, sum(x["LocA"] * x["TP"] for x in lst)) / np.maximum(1e-10, r["TP"])
    return _hota_final(r)


def hota_resumo(r):
    return {"HOTA": float(r["HOTA"].mean()), "DetA": float(r["DetA"].mean()), "AssA": float(r["AssA"].mean()),
            "DetRe": float(r["DetRe"].mean()), "DetPr": float(r["DetPr"].mean()), "AssRe": float(r["AssRe"].mean()),
            "AssPr": float(r["AssPr"].mean()), "LocA": float(r["LocA"].mean()), "HOTA(0)": float(r["HOTA"][0]),
            "HOTA_alfa50": float(r["HOTA"][9])}


# ----------------------------------------------------------------------------------------------- CLEAR (MOTA)
def clear(seq, thr=0.5, pares=None):
    """CLEAR MOT. Se `pares` for um set, recebe os pares (id_gt, id_trk) casados em algum quadro."""
    ng = seq["n_gt_ids"]
    r = dict(TP=0, FN=0, FP=0, IDSW=0, MOTP_soma=0.0, MT=0, PT=0, ML=0, Frag=0, n_gt_ids=ng, n_gt_dets=seq["n_gt_dets"])
    if seq["n_trk_dets"] == 0:
        r["FN"] = seq["n_gt_dets"]; r["ML"] = ng
        return r
    if seq["n_gt_dets"] == 0:
        r["FP"] = seq["n_trk_dets"]
        return r
    gcount = np.zeros(ng); gmatched = np.zeros(ng); gfrag = np.zeros(ng)
    prev = np.full(ng, np.nan); prev_t = np.full(ng, np.nan)
    for g, k, s in zip(seq["gt_ids"], seq["trk_ids"], seq["sim"]):
        if len(g) == 0:
            r["FP"] += len(k); continue
        if len(k) == 0:
            r["FN"] += len(g); gcount[g] += 1; continue
        score = 1000 * (k[None, :] == prev_t[g[:, None]]) + s
        score[s < thr - EPS] = 0
        mr, mc = linear_sum_assignment(-score)
        ok = score[mr, mc] > 0 + EPS
        mr, mc = mr[ok], mc[ok]
        mg, mt = g[mr], k[mc]
        if pares is not None:
            pares.update(zip(mg.tolist(), mt.tolist()))
        pm = prev[mg]
        r["IDSW"] += int(np.sum(~np.isnan(pm) & (mt != pm)))
        gcount[g] += 1; gmatched[mg] += 1
        not_prev = np.isnan(prev_t)
        prev[mg] = mt
        prev_t[:] = np.nan; prev_t[mg] = mt
        gfrag += not_prev & ~np.isnan(prev_t)
        n = len(mg)
        r["TP"] += n; r["FN"] += len(g) - n; r["FP"] += len(k) - n
        if n:
            r["MOTP_soma"] += float(s[mr, mc].sum())
    ratio = gmatched[gcount > 0] / gcount[gcount > 0]
    r["MT"] = int(np.sum(ratio > 0.8)); r["PT"] = int(np.sum(ratio >= 0.2)) - r["MT"]; r["ML"] = ng - r["MT"] - r["PT"]
    r["Frag"] = int(np.sum(np.subtract(gfrag[gfrag > 0], 1)))
    return r


def clear_combinar(lst):
    r = {k: sum(x[k] for x in lst) for k in ("TP", "FN", "FP", "IDSW", "MOTP_soma", "MT", "PT", "ML", "Frag", "n_gt_ids", "n_gt_dets")}
    return r


def clear_resumo(r):
    return {"MOTA": (r["TP"] - r["FP"] - r["IDSW"]) / max(1, r["TP"] + r["FN"]), "MOTP": r["MOTP_soma"] / max(1, r["TP"]),
            "IDSW": int(r["IDSW"]), "CLR_TP": int(r["TP"]), "CLR_FN": int(r["FN"]), "CLR_FP": int(r["FP"]),
            "MT": int(r["MT"]), "PT": int(r["PT"]), "ML": int(r["ML"]), "Frag": int(r["Frag"]),
            "CLR_Re": r["TP"] / max(1, r["TP"] + r["FN"]), "CLR_Pr": r["TP"] / max(1, r["TP"] + r["FP"])}


# ----------------------------------------------------------------------------------------------- Identity (IDF1)
def identity(seq, thr=0.5):
    ng, nt = seq["n_gt_ids"], seq["n_trk_ids"]
    r = dict(IDTP=0, IDFN=seq["n_gt_dets"], IDFP=seq["n_trk_dets"])
    if seq["n_trk_dets"] == 0 or seq["n_gt_dets"] == 0:
        return r
    pot = np.zeros((ng, nt))
    for g, k, s in zip(seq["gt_ids"], seq["trk_ids"], seq["sim"]):
        if len(g) and len(k):
            mg, mt = np.nonzero(s >= thr - EPS)
            np.add.at(pot, (g[mg], k[mt]), 1)
    # minimizar IDFN+IDFP (TrackEval) == maximizar a soma de coincidencias do emparelhamento global
    mr, mc = linear_sum_assignment(-pot)
    idtp = float(pot[mr, mc].sum())
    r["IDTP"] = idtp; r["IDFN"] = seq["n_gt_dets"] - idtp; r["IDFP"] = seq["n_trk_dets"] - idtp
    return r


def identity_combinar(lst):
    return {k: sum(x[k] for x in lst) for k in ("IDTP", "IDFN", "IDFP")}


def identity_resumo(r):
    return {"IDF1": r["IDTP"] / max(1, r["IDTP"] + 0.5 * r["IDFP"] + 0.5 * r["IDFN"]),
            "IDR": r["IDTP"] / max(1, r["IDTP"] + r["IDFN"]), "IDP": r["IDTP"] / max(1, r["IDTP"] + r["IDFP"])}


def avaliar(seq, pares=None):
    """Todas as metricas de uma sequencia (dicionarios brutos, para combinar depois)."""
    return {"hota": hota(seq), "clear": clear(seq, pares=pares), "identity": identity(seq)}


def resumir(lst_brutos):
    h = hota_combinar([x["hota"] for x in lst_brutos])
    c = clear_combinar([x["clear"] for x in lst_brutos])
    i = identity_combinar([x["identity"] for x in lst_brutos])
    return {**hota_resumo(h), **clear_resumo(c), **identity_resumo(i)}


# ----------------------------------------------------------------------------------------------- contagem
def erros_contagem(pred, gt):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    e = pred - gt
    nz = gt > 0
    return {"MAE": float(np.abs(e).mean()) if len(e) else float("nan"),
            "RMSE": float(np.sqrt((e ** 2).mean())) if len(e) else float("nan"),
            "MAPE_pct": float(100 * np.mean(np.abs(e[nz]) / gt[nz])) if nz.any() else float("nan"),
            "vies_medio": float(e.mean()) if len(e) else float("nan"),
            "n_quadros": int(len(gt)), "gt_medio": float(gt.mean()) if len(gt) else float("nan"),
            "pred_medio": float(pred.mean()) if len(pred) else float("nan")}
