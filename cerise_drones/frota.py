"""Modelo da frota: parametros de voo, camera e bateria de cada drone."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class Drone:
    id: str
    velocidade_ms: float = 10.0        # velocidade de cruzeiro durante a varredura
    autonomia_min: float = 20.0        # tempo de voo com bateria cheia (pairar/cruzeiro)
    hfov_graus: float = 70.0           # campo de visao horizontal (transversal ao voo)
    vfov_graus: float = 55.0           # campo de visao vertical (ao longo do voo)
    altitude_m: float = 50.0           # altitude relativa ao ponto de decolagem
    partida_xy: tuple | None = None    # base de decolagem em metros UTM (preenchida pelo cenario)
    reserva_bateria: float = 0.20      # fracao da bateria que nunca e usada (pouso seguro)
    vel_subida_ms: float = 3.0
    vel_descida_ms: float = 2.0
    tempo_curva_s: float = 4.0         # penalidade por curva (desaceleracao + giro + aceleracao)
    tempo_troca_bateria_s: float = 300.0
    tipo: str = "A"
    rotulo: str = ""

    # --- camera / pegada no solo (terreno plano, camera em nadir) ---
    @property
    def largura_pegada_m(self) -> float:
        return 2.0 * self.altitude_m * math.tan(math.radians(self.hfov_graus) / 2.0)

    @property
    def comprimento_pegada_m(self) -> float:
        return 2.0 * self.altitude_m * math.tan(math.radians(self.vfov_graus) / 2.0)

    def espacamento_m(self, sobreposicao_lateral: float) -> float:
        """Distancia entre linhas de varredura adjacentes."""
        return self.largura_pegada_m * (1.0 - sobreposicao_lateral)

    def distancia_disparo_m(self, sobreposicao_frontal: float) -> float:
        """Distancia entre fotos consecutivas (MAV_CMD_DO_SET_CAM_TRIGG_DIST)."""
        return self.comprimento_pegada_m * (1.0 - sobreposicao_frontal)

    def taxa_cobertura_m2s(self, sobreposicao_lateral: float) -> float:
        return self.velocidade_ms * self.espacamento_m(sobreposicao_lateral)

    @property
    def autonomia_util_s(self) -> float:
        return self.autonomia_min * 60.0 * (1.0 - self.reserva_bateria)

    def capacidade_area_m2(self, sobreposicao_lateral: float) -> float:
        """Area varrida com uma bateria (ignorando transito e curvas)."""
        return self.taxa_cobertura_m2s(sobreposicao_lateral) * self.autonomia_util_s

    def para_dict(self) -> dict:
        d = asdict(self)
        d["largura_pegada_m"] = round(self.largura_pegada_m, 2)
        d["comprimento_pegada_m"] = round(self.comprimento_pegada_m, 2)
        return d


# Tipos de referencia usados nos experimentos (parametros representativos de multirrotores de
# pequeno porte; nao correspondem a um modelo comercial especifico).
TIPOS = {
    "A": dict(velocidade_ms=10.0, autonomia_min=20.0, hfov_graus=70.0, vfov_graus=55.0, rotulo="quad. medio"),
    "B": dict(velocidade_ms=6.0, autonomia_min=30.0, hfov_graus=70.0, vfov_graus=55.0, rotulo="quad. lento, bateria maior"),
    "C": dict(velocidade_ms=14.0, autonomia_min=25.0, hfov_graus=82.0, vfov_graus=52.0, rotulo="quad. rapido, grande-angular"),
}


def criar_frota(tipos: list[str], partidas_xy: list[tuple], altitude_m: float = 50.0,
                **extra) -> list[Drone]:
    frota = []
    for i, (t, p) in enumerate(zip(tipos, partidas_xy)):
        params = dict(TIPOS[t])
        params.update(extra)
        frota.append(Drone(id=f"D{i + 1}", tipo=t, altitude_m=altitude_m, partida_xy=tuple(p), **params))
    return frota


def pesos_frota(frota: list[Drone], modo: str, sobreposicao_lateral: float) -> list[float]:
    """Pesos (fracao da area) para a particao ponderada.

    'taxa'       : v * espacamento (m2/s) -> iguala o tempo de varredura (minimiza o makespan)
    'capacidade' : v * espacamento * autonomia util (m2 por bateria) -> iguala o uso de bateria
    'igual'      : todos iguais
    """
    if modo == "igual":
        w = [1.0] * len(frota)
    elif modo == "capacidade":
        w = [d.capacidade_area_m2(sobreposicao_lateral) for d in frota]
    elif modo == "taxa":
        w = [d.taxa_cobertura_m2s(sobreposicao_lateral) for d in frota]
    else:
        raise ValueError(modo)
    s = sum(w)
    return [x / s for x in w]
