"""Pipeline de contagem: quadros + pose -> detector (YOLO11s ajustado, FP32) -> projecao no chao ->
rastreio por drone -> posse da subarea + deduplicacao espaco-temporal (inclusive entre drones na faixa
de fronteira) -> contagem global unica.

Estrategias reportadas (para comparacao):
- soma_deteccoes          : soma de todas as deteccoes de todos os quadros (limite superior ingenuo)
- faixas_sem_posse        : soma das faixas confirmadas de cada drone (sem posse nem dedup entre drones)
- cerise_drones_agregacao : cerise_drones.agregacao.contar() aplicada as posicoes representativas das
                            faixas (posse + dedup por raio + faixa de fronteira; supoe pessoas paradas)
- proposto                : faixas com posse (ou na faixa de fronteira) agrupadas por compatibilidade
                            espaco-temporal (sobreposicao no tempo: distancia mediana <= raio; sem
                            sobreposicao: distancia <= raio + v_assoc * intervalo, intervalo <= gap_max)
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import shapely

from .camera import pixel_para_chao
from .rastreador import Faixa, RastreadorChao


@dataclass
class ParamsContagem:
    ponto_ref: str = "centro"          # "centro" da caixa a altura_ref_m, ou "base" (z = 0; camera obliqua)
    altura_ref_m: float = 0.85
    somente_cobertura: bool = True     # usa so quadros em pernas de cobertura (camera "ativa" do plano)
    passo_quadros: int = 1
    margem_area_m: float = 2.0         # deteccoes projetadas fora da area (+ margem) sao descartadas
    dist_max_m: float = 2.0            # rastreio
    vel_max_ms: float = 2.5
    min_hits: int = 3
    max_idade_s: float = 2.0
    raio_m: float = 2.0                # agregacao
    banda_m: float = 6.0
    v_parado_ms: float = 0.3           # abaixo disso a faixa e tratada como pessoa parada
    v_assoc_ms: float = 0.3            # incerteza de velocidade na extrapolacao (pessoas andando)
    v_assoc_parado_ms: float = 0.1     # idem para faixas quase paradas
    gap_max_s: float = 120.0           # intervalo maximo para reassociar pessoas paradas
    gap_max_andando_s: float = 60.0    # idem para pessoas andando (escolhido na sensibilidade sem render)


# ------------------------------------------------------------------------------------------------
# Detectores
# ------------------------------------------------------------------------------------------------
class DetectorYOLO:
    """YOLO (ultralytics) em FP32 (FP16 da NaN no GB10 com o PyTorch do NGC 26.05)."""

    def __init__(self, pesos, imgsz: int = 1280, conf: float = 0.175, iou: float = 0.5, device=0, max_det: int = 1000):
        from ultralytics import YOLO
        self.m = YOLO(str(pesos))
        self.imgsz, self.conf, self.iou, self.device, self.max_det = imgsz, conf, iou, device, max_det
        nomes = {int(k): str(v).lower() for k, v in self.m.names.items()}
        alvo = [k for k, v in nomes.items() if v in ("pessoa", "person", "pedestrian", "people")]
        self.classes = alvo if len(nomes) > 1 else None
        self.nomes = nomes

    def __call__(self, imagem):
        r = self.m.predict(str(imagem) if isinstance(imagem, Path) else imagem, imgsz=self.imgsz, conf=self.conf,
                           iou=self.iou, half=False, device=self.device, verbose=False, max_det=self.max_det,
                           classes=self.classes)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float)


class DetectorOraculo:
    """Deteccoes sinteticas a partir das caixas de gabarito (testes e analise do erro da agregacao
    isolado do erro do detector): revocacao p_det, ruido gaussiano nos cantos, falsos positivos
    Poisson por quadro em posicoes aleatorias."""

    def __init__(self, caixas_gt: dict, largura: int, altura: int, p_det: float = 1.0, sigma_px: float = 0.0,
                 fp_por_quadro: float = 0.0, oclusao_max: float = 0.9, seed: int = 0):
        self.gt, self.W, self.H = caixas_gt, largura, altura
        self.p_det, self.sigma, self.fp, self.ocl = p_det, sigma_px, fp_por_quadro, oclusao_max
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame: int):
        ids, bx, oc = self.gt.get(frame, ([], np.zeros((0, 4)), np.zeros(0)))
        bx = np.asarray(bx, float).reshape(-1, 4)
        m = (np.asarray(oc) <= self.ocl) & (self.rng.random(len(bx)) < self.p_det)
        b = bx[m] + (self.rng.normal(0, self.sigma, (int(m.sum()), 4)) if self.sigma > 0 else 0)
        nfp = self.rng.poisson(self.fp) if self.fp > 0 else 0
        if nfp:
            c = self.rng.uniform([0, 0], [self.W, self.H], (nfp, 2))
            s = self.rng.uniform(6, 16, (nfp, 1))
            b = np.vstack([b, np.hstack([c - s / 2, c + s / 2])])
        return b, np.full(len(b), 0.9)


# ------------------------------------------------------------------------------------------------
# Etapas
# ------------------------------------------------------------------------------------------------
def caixas_para_chao(caixas, pose, K, p: ParamsContagem) -> np.ndarray:
    caixas = np.asarray(caixas, float).reshape(-1, 4)
    if p.ponto_ref == "base":
        uv = np.column_stack([(caixas[:, 0] + caixas[:, 2]) / 2, caixas[:, 3]])
        return pixel_para_chao(uv, pose, K, 0.0)[:, :2]
    uv = np.column_stack([(caixas[:, 0] + caixas[:, 2]) / 2, (caixas[:, 1] + caixas[:, 3]) / 2])
    return pixel_para_chao(uv, pose, K, p.altura_ref_m)[:, :2]


def rastrear_camera(quadros, K, detectar, area, p: ParamsContagem, por_frame: bool = False):
    """quadros: lista de dados.Quadro; detectar(quadro) -> (caixas, scores).
    Retorna (faixas confirmadas, log por quadro {frame: (caixas, scores, xy)})."""
    rast = RastreadorChao(p.dist_max_m, p.vel_max_ms, p.min_hits, p.max_idade_s)
    area_m = area.buffer(p.margem_area_m) if area is not None else None
    if area_m is not None:
        shapely.prepare(area_m)
    log = {}
    n_det = 0
    for i, q in enumerate(quadros):
        if p.somente_cobertura and not q.cobre:
            continue
        if i % p.passo_quadros:
            continue
        caixas, scores = detectar(q)
        caixas = np.asarray(caixas, float).reshape(-1, 4)
        scores = np.asarray(scores, float).reshape(-1)
        xy = caixas_para_chao(caixas, q.pose, K, p)
        ok = np.isfinite(xy).all(axis=1)
        if area_m is not None and len(xy):
            ok &= shapely.contains_xy(area_m, np.nan_to_num(xy[:, 0]), np.nan_to_num(xy[:, 1]))
        rast.atualizar(q.t, xy[ok], scores[ok], q.frame, caixas[ok])
        n_det += int(ok.sum())
        if por_frame:
            log[q.frame] = (caixas[ok], scores[ok], xy[ok])
    return rast.finalizar(), log, n_det


def _velocidade(f: Faixa, fim: bool, n: int = 10) -> np.ndarray:
    """Velocidade (m/s) por minimos quadrados nas n ultimas (fim) ou n primeiras deteccoes."""
    t = np.asarray(f.t, float); xy = np.asarray(f.xy, float)
    sl = slice(-n, None) if fim else slice(0, n)
    t, xy = t[sl], xy[sl]
    if len(t) < 3 or t[-1] - t[0] < 0.5:
        return np.zeros(2)
    A = np.column_stack([t - t.mean(), np.ones(len(t))])
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return coef[0]


def _custo(a: Faixa, b: Faixa, p: ParamsContagem):
    """Custo normalizado (<= 1 => compativeis) ou None. Sobreposicao no tempo: distancia mediana nos
    instantes comuns <= raio. Sem sobreposicao: extrapolacao a velocidade constante da faixa anterior
    (e retro-extrapolacao da posterior), tolerancia raio + v_assoc * intervalo; faixas quase paradas
    (< v_parado) usam v_assoc_parado e aceitam intervalos ate gap_max; andando, ate gap_max_andando."""
    t0, t1 = max(a.t_ini, b.t_ini), min(a.t_fim, b.t_fim)
    if t1 >= t0:
        tt = np.linspace(t0, t1, max(2, int((t1 - t0) * 2) + 1))
        d = float(np.median(np.linalg.norm(a.posicao_em(tt) - b.posicao_em(tt), axis=1)))
        return d / p.raio_m if d <= p.raio_m else None
    e, l = (a, b) if a.t_fim < b.t_ini else (b, a)
    gap = l.t_ini - e.t_fim
    ve, vl = _velocidade(e, True), _velocidade(l, False)
    parado = np.linalg.norm(ve) < p.v_parado_ms and np.linalg.norm(vl) < p.v_parado_ms
    if gap > (p.gap_max_s if parado else p.gap_max_andando_s):
        return None
    pe, pl = np.asarray(e.xy[-1], float), np.asarray(l.xy[0], float)
    d = min(float(np.linalg.norm(pe + ve * gap - pl)), float(np.linalg.norm(pl - vl * gap - pe)))
    tol = p.raio_m + (p.v_assoc_parado_ms if parado else p.v_assoc_ms) * gap
    return d / tol if d <= tol else None


def _sobrepoem(a: Faixa, b: Faixa) -> bool:
    return min(a.t_fim, b.t_fim) >= max(a.t_ini, b.t_ini)


def agregar(faixas_por_drone: dict, ids_drones: list, regioes: list, p: ParamsContagem) -> dict:
    """faixas_por_drone: {drone_id: [Faixa]} ; regioes: subareas (shapely, ENU) na ordem de ids_drones."""
    itens = []  # (drone_idx, faixa)
    for i, d in enumerate(ids_drones):
        for f in faixas_por_drone.get(d, []):
            itens.append((i, f))
    out = {"faixas_sem_posse": len(itens)}
    if not itens:
        out.update(proposto=0, clusters=[], descartadas_posse=0)
        return out
    rep = np.array([f.posicao_representativa() for _, f in itens])
    drone = np.array([i for i, _ in itens])
    # reaproveita a agregacao do cerise_drones (posicoes estaticas)
    try:
        raiz = Path(__file__).resolve().parents[2]
        if str(raiz) not in sys.path:
            sys.path.insert(0, str(raiz))
        from cerise_drones.agregacao import contar as contar_cd
        out["cerise_drones_agregacao"] = contar_cd(rep, drone, regioes, raio_m=p.raio_m, banda_m=p.banda_m)
    except Exception as e:  # pacote ausente no ambiente
        out["cerise_drones_agregacao"] = {"erro": repr(e)}
    # posse
    dono = np.full(len(itens), -1)
    for i, g in enumerate(regioes):
        shapely.prepare(g)
        dono[shapely.contains_xy(g, rep[:, 0], rep[:, 1])] = i
    front = shapely.union_all([g.boundary for g in regioes])
    banda = front.buffer(p.banda_m)
    shapely.prepare(banda)
    na_banda = shapely.contains_xy(banda, rep[:, 0], rep[:, 1])
    manter = (dono == drone) | na_banda
    idx = np.nonzero(manter)[0]
    out["descartadas_posse"] = int((~manter).sum())
    # pares compativeis (custo <= 1)
    pares = []
    alcance = p.raio_m + max(p.v_assoc_ms * p.gap_max_andando_s + 2.0 * p.gap_max_andando_s, 0) + 5
    for a_i in range(len(idx)):
        a = idx[a_i]
        for b_i in range(a_i + 1, len(idx)):
            b = idx[b_i]
            mesmo = drone[a] == drone[b]
            fa, fb = itens[a][1], itens[b][1]
            if mesmo and _sobrepoem(fa, fb):
                continue  # mesmo drone ao mesmo tempo: pessoas diferentes
            if not mesmo and not (na_banda[a] or na_banda[b]):
                continue
            if np.linalg.norm(rep[a] - rep[b]) > alcance:
                continue
            c = _custo(fa, fb, p)
            if c is not None:
                pares.append((c, a_i, b_i))
    # fusao gulosa por custo crescente com restricao "nao pode ligar": um cluster nunca junta duas
    # faixas simultaneas do mesmo drone, nem faixas simultaneas de drones diferentes que estejam longe
    n = len(idx)
    pai = list(range(n))
    membros = {i: [i] for i in range(n)}

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    def conflito(A, B):
        for x in A:
            fx = itens[idx[x]][1]
            for y in B:
                fy = itens[idx[y]][1]
                if not _sobrepoem(fx, fy):
                    continue
                if drone[idx[x]] == drone[idx[y]] or _custo(fx, fy, p) is None:
                    return True
        return False

    for c, a_i, b_i in sorted(pares):
        ra, rb = raiz(a_i), raiz(b_i)
        if ra == rb or conflito(membros[ra], membros[rb]):
            continue
        pai[rb] = ra
        membros[ra] += membros.pop(rb)
    clusters = [[(ids_drones[drone[idx[j]]], itens[idx[j]][1].id) for j in ms] for ms in membros.values()]
    out.update(proposto=len(clusters), clusters=clusters, frac_na_banda=float(na_banda.mean()),
               pares_compativeis=len(pares))
    return out


def contar_execucao(execucao, detector_por_camera, area, regioes, ids_drones, p: ParamsContagem | None = None,
                    por_frame: bool = True) -> dict:
    """detector_por_camera(cam) -> funcao detectar(quadro) -> (caixas, scores)."""
    p = p or ParamsContagem()
    faixas, logs, ndet, tempos = {}, {}, {}, {}
    for cam in ids_drones:
        t0 = time.perf_counter()
        quadros = execucao.quadros(cam)
        K = execucao.intrinsecos(cam)
        faixas[cam], logs[cam], ndet[cam] = rastrear_camera(quadros, K, detector_por_camera(cam), area, p, por_frame)
        tempos[cam] = time.perf_counter() - t0
    ag = agregar(faixas, ids_drones, regioes, p)
    ag["soma_deteccoes"] = int(sum(ndet.values()))
    return {"parametros": asdict(p), "agregacao": ag, "faixas": faixas, "logs": logs, "tempo_s": tempos}
