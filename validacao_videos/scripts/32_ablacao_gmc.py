"""Ablacao da compensacao de movimento de camera (GMC) do BoT-SORT: mesma configuracao escolhida no val, com
gmc_method = sparseOptFlow (padrao) e = none, no val e no test-dev, para a combinacao principal e a referencia.
Separa o efeito da GMC das demais diferencas BoT-SORT x ByteTrack (Kalman em xywh, limiares).
Saida: resultados/ablacao_gmc.json
"""
import importlib
import json
from multiprocessing import Pool

r30 = importlib.import_module("30_rastrear")
sel = json.load(open(f"{r30.RES}/selecao_val.json"))
meta = json.load(open(f"{r30.RES}/evidencias_meta.json"))

if __name__ == "__main__":
    tarefas = []
    for combo in (meta["principal"], meta["referencia"]):
        v, n = combo.split("|")
        cfg = sel["escolha"][combo]["cfg"]
        for gmc in ("sparseOptFlow", "none"):
            for split in ("val", "test-dev"):
                for s in r30.SEQS[split]:
                    tarefas.append((v, n, gmc, {**cfg, "gmc_method": gmc}, split, s, False))
    agg = {}
    with Pool(4) as p:
        for r in p.imap_unordered(r30.tarefa, tarefas):
            agg.setdefault(f"{r['variante']}|{r['rastreador']}|gmc={r['cfg_idx']}|{r['split']}", []).append(r)
    out = {}
    for k, rs in sorted(agg.items()):
        m = r30.resumo_lista(rs)
        out[k] = {x: m[x] for x in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")}
        out[k]["unicos_pred_L1"] = sum(r["unicos_pred"][1] for r in rs)
        out[k]["unicos_gt"] = sum(r["unicos_gt"] for r in rs)
        print(k, json.dumps(out[k]), flush=True)
    json.dump(out, open(f"{r30.RES}/ablacao_gmc.json", "w"), indent=1)
    print("FIM_ABLACAO", flush=True)
