"""Teste rapido antes da grade: roda 30_rastrear.tarefa numa sequencia do val com a configuracao padrao,
para cada rastreador e para o detector ajustado, e imprime metricas e tempos."""
import importlib
import json
import time

r30 = importlib.import_module("30_rastrear")
seq = "uav0000086_00000_v"
for n in r30.RASTREADORES:
    for v in ("ft960", "coco960"):
        t0 = time.time()
        r = r30.tarefa((v, n, -1, r30.PADRAO, "val", seq, False))
        x = r30.sem_arrays(r)
        print(json.dumps({k: x[k] for k in ("variante", "rastreador", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "unicos_gt",
                                            "unicos_pred", "decomp_L1", "tempo_rastreio_s")}), f"total {time.time()-t0:.1f}s", flush=True)
print("FIM_TESTE", flush=True)
