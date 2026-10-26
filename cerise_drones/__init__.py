"""cerise_drones - segmentacao da area de atuacao entre drones, planejamento de cobertura,
colaboracao (realocacao dinamica) e protocolo de mensagens LoRa. Projeto CERISE/EMC/UFG 2026."""
from .cobertura import Navegador, Rota, planejar_regiao, simular_missao
from .frota import TIPOS, Drone, criar_frota, pesos_frota
from .geo import AreaMissao, Projetor
from .particao import ResultadoParticao, particionar
from .planejador import PlanoMissao, planejar_missao

__version__ = "0.1.0"
__all__ = ["AreaMissao", "Projetor", "Drone", "TIPOS", "criar_frota", "pesos_frota", "particionar",
           "ResultadoParticao", "planejar_regiao", "simular_missao", "Navegador", "Rota",
           "planejar_missao", "PlanoMissao"]
