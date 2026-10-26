"""Sensibilidade (diagnostico a posteriori) da contagem unica ao limiar de criacao/associacao de trilhas do BoT-SORT
(track_high_thresh = new_track_thresh) e ao limiar de confianca do rastreador no solo, sobre o cache de deteccoes
de uma execucao ja contada por scripts/contar.py. NAO e usada para escolher a configuracao principal (que vem da
validacao em videos reais); serve para medir quanto da subcontagem se deve a calibracao das pontuacoes no dominio
simulado.

  python scripts/varredura_botsort.py --execucao saidas/run01 --cenario cenario/reduzido \
      --cache resultados/contagem_run01_cache_det.npz --saida resultados/varredura_run01.json
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(SIM.parent))
import numpy as np  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from cerise_sim.avaliacao import avaliar_contagem  # noqa: E402
from cerise_sim.botsort_chao import ParamsBotSort, carregar_cache, contar_execucao_botsort  # noqa: E402
from cerise_sim.contagem import ParamsContagem, contar_execucao  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402


def poli(lista):
    ps = [Polygon(c) for c in lista]
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)


def resumo(av, res):
    return {k: av[k] for k in ("contagem_proposta", "pessoas_vistas_gabarito", "erro_pct_vs_vistas", "clusters_corretos_unicos",
                               "clusters_duplicados", "clusters_falsos", "pessoas_perdidas")} | \
        {"faixas": sum(len(v) for v in res["faixas"].values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    cen = json.loads((Path(a.cenario) / "cenario.json").read_text(encoding="utf-8"))
    ex = Execucao(a.execucao)
    ids = [d for d in cen["regioes_enu"] if d in ex.cameras("drone")]
    area = poli(cen["area_enu"]); regioes = [poli(cen["regioes_enu"][d]) for d in ids]
    gts = {c: ex.caixas_gt(c) for c in ids}
    dets, gmcs = carregar_cache(a.cache)
    p = ParamsContagem()
    out = {"botsort": [], "chao": [], "parametros_contagem": asdict(p)}
    for fuse in (True, False):
        for h in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            pb = ParamsBotSort(track_high_thresh=h, new_track_thresh=h, track_low_thresh=min(0.05, h / 2), fuse_score=fuse)
            t0 = time.time()
            res = contar_execucao_botsort(ex, dets, gmcs, area, regioes, ids, p, pb)
            av = avaliar_contagem(res, gts, len(ex.pessoas()))
            r = {"track_high_new": h, "track_low": pb.track_low_thresh, "fuse_score": fuse, **resumo(av, res),
                 "s": round(time.time() - t0, 1)}
            out["botsort"].append(r)
            print("botsort", r, flush=True)
    for cf in (0.05, 0.10, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40):
        def fabrica(cam, cf=cf):
            def det(q):
                c, s = dets[cam].get(q.frame, (np.zeros((0, 4)), np.zeros(0)))
                return c[s >= cf], s[s >= cf]
            return det
        res = contar_execucao(ex, fabrica, area, regioes, ids, p)
        av = avaliar_contagem(res, gts, len(ex.pessoas()))
        r = {"conf": cf, **resumo(av, res)}
        out["chao"].append(r)
        print("chao", r, flush=True)
    Path(a.saida).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
