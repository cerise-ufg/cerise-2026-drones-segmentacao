"""Plano de pessoas com verdade-terreno (gabarito) de posicoes ao longo do tempo.

Tipos de comportamento (proporcoes aproximadas de N):
- parado          ~12%: em pe no mesmo lugar o tempo todo (rumo aleatorio)
- grupo_parado    ~15%: rodas de conversa de 3-5 pessoas voltadas para o centro
- grupo_andando   ~18%: grupos de 2-4 pessoas andando juntas (deslocamentos fixos ao lider)
- cruza_fronteira ~10%: caminham de um lado para o outro atravessando a fronteira entre subareas
                        (teste de dupla contagem entre drones)
- caminhante      resto: andam entre pontos aleatorios ou ao longo dos caminhos, com paradas.
Cada pessoa tem quadros-chave (t, x, y, yaw) com interpolacao linear (paradas = dois quadros-chave na
mesma posicao). O mesmo plano gera os comandos do omni.anim.people (GoTo/Idle/LookAround) para o
Isaac Sim e as trajetorias de <actor> do Gazebo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


@dataclass
class Pessoa:
    id: str
    tipo: str
    grupo: int
    velocidade_ms: float
    kf: list = field(default_factory=list)        # [(t, x, y, yaw_graus)]
    comandos: list = field(default_factory=list)  # comandos omni.anim.people (sem o nome)

    def arrays(self):
        a = np.array(self.kf, float)
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    def posicao(self, tt):
        t, x, y, w = self.arrays()
        tt = np.clip(np.atleast_1d(tt), t[0], t[-1])
        return np.interp(tt, t, x), np.interp(tt, t, y), np.interp(tt, t, w)

    def para_dict(self) -> dict:
        return {"id": self.id, "tipo": self.tipo, "grupo": self.grupo, "velocidade_ms": round(self.velocidade_ms, 3),
                "kf": [[round(v, 3) for v in k] for k in self.kf], "comandos": self.comandos}


def _rumo(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _desenrolar(ant, novo):
    return ant + ((novo - ant + 180.0) % 360.0 - 180.0)


class _Caminhada:
    """Acumula quadros-chave e comandos de uma caminhada."""

    def __init__(self, inicio, yaw, vel):
        self.t = 0.0
        self.p = np.asarray(inicio, float)
        self.yaw = yaw
        self.vel = vel
        self.kf = [(0.0, self.p[0], self.p[1], yaw)]
        self.cmd = []

    def ir(self, q, comando=True):
        q = np.asarray(q, float)
        d = float(np.linalg.norm(q - self.p))
        if d < 1e-3:
            return
        novo = _desenrolar(self.yaw, _rumo(self.p, q))
        # giro rapido no lugar (0,4 s) antes de andar
        self.t += 0.4
        self.kf.append((self.t, self.p[0], self.p[1], novo))
        self.yaw = novo
        self.t += d / self.vel
        self.p = q
        self.kf.append((self.t, q[0], q[1], novo))
        if comando:
            self.cmd.append(f"GoTo {q[0]:.2f} {q[1]:.2f} 0.0 _")

    def parar(self, dur, olhar=False):
        self.t += dur
        self.kf.append((self.t, self.p[0], self.p[1], self.yaw))
        self.cmd.append(f"{'LookAround' if olhar else 'Idle'} {dur:.1f}")


def _ponto_dentro(rng, area_in: Polygon, tent=1000):
    minx, miny, maxx, maxy = area_in.bounds
    for _ in range(tent):
        p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if area_in.contains(Point(p)):
            return np.array(p)
    raise RuntimeError("area muito pequena")


def _proximo_destino(rng, area_in, p, dmin=20, dmax=90):
    for _ in range(80):
        ang = rng.uniform(-math.pi, math.pi)
        d = rng.uniform(dmin, dmax)
        q = p + d * np.array([math.cos(ang), math.sin(ang)])
        if area_in.contains(LineString([tuple(p), tuple(q)])):
            return q
    return None


def _caminhar_livre(rng, area_in, inicio, vel, duracao, p_pausa=0.45):
    c = _Caminhada(inicio, rng.uniform(-180, 180), vel)
    while c.t < duracao:
        q = _proximo_destino(rng, area_in, c.p)
        if q is None:
            c.parar(5.0)
            continue
        c.ir(q)
        if rng.random() < p_pausa:
            c.parar(float(rng.uniform(3, 25)), olhar=rng.random() < 0.4)
    return c


def _caminhar_no_caminho(rng, area_in, linha: LineString, vel, duracao):
    s = rng.uniform(0, linha.length)
    lat = rng.uniform(-0.6, 0.6)

    def ponto(si):
        si = float(np.clip(si, 0, linha.length))
        a = linha.interpolate(si); b = linha.interpolate(min(linha.length, si + 0.5))
        if b.distance(a) < 1e-6:
            b = a; a = linha.interpolate(max(0.0, si - 0.5))
        d = np.array([b.x - a.x, b.y - a.y]); d /= max(np.linalg.norm(d), 1e-9)
        n = np.array([-d[1], d[0]])
        return np.array([linha.interpolate(si).x, linha.interpolate(si).y]) + lat * n

    c = _Caminhada(ponto(s), rng.uniform(-180, 180), vel)
    sentido = rng.choice([-1, 1])
    while c.t < duracao:
        alvo = s + sentido * rng.uniform(30, 120)
        if alvo <= 0 or alvo >= linha.length:
            sentido *= -1
            alvo = float(np.clip(alvo, 0, linha.length))
        passos = np.linspace(s, alvo, max(2, int(abs(alvo - s) / 10) + 1))[1:]
        for si in passos:
            q = ponto(si)
            if area_in.contains(Point(q)):
                c.ir(q, comando=(si == passos[-1]))
        # o omni.anim.people recebe so o destino final de cada trecho (o planejador de rota dele
        # segue a navmesh); os vertices intermediarios ficam no gabarito do modo cinematico/Gazebo
        s = alvo
        if rng.random() < 0.35:
            c.parar(float(rng.uniform(3, 15)))
    return c


def gerar_pessoas(area: Polygon, n: int, duracao_s: float, seed: int = 0, caminhos=None,
                  fronteiras=None, margem_m: float = 3.0) -> list[Pessoa]:
    rng = np.random.default_rng(seed)
    area_in = area.buffer(-margem_m)
    linhas = []
    for c in (caminhos or []):
        L = LineString(c["linha"]).intersection(area.buffer(-1.5))
        for g in getattr(L, "geoms", [L]):
            if isinstance(g, LineString) and g.length > 40:
                linhas.append(g)
    n_parado = int(round(0.12 * n))
    n_gparado = int(round(0.15 * n))
    n_gandando = int(round(0.18 * n))
    n_fronteira = int(round(0.10 * n)) if fronteiras is not None and not fronteiras.is_empty else 0
    pessoas: list[Pessoa] = []
    grupo = 0

    def novo_id():
        return f"P{len(pessoas):03d}"

    # rodas paradas
    restantes = n_gparado
    while restantes > 0:
        k = int(min(restantes, rng.integers(3, 6)))
        c = _ponto_dentro(rng, area.buffer(-margem_m - 2))
        r = 0.55 + 0.12 * k
        a0 = rng.uniform(0, 2 * math.pi)
        for j in range(k):
            a = a0 + 2 * math.pi * j / k + rng.normal(0, 0.15)
            p = c + r * np.array([math.cos(a), math.sin(a)])
            yaw = math.degrees(math.atan2(c[1] - p[1], c[0] - p[0]))
            pessoas.append(Pessoa(novo_id(), "grupo_parado", grupo, 0.0,
                                  [(0.0, p[0], p[1], yaw), (duracao_s, p[0], p[1], yaw)],
                                  [f"Idle {duracao_s:.1f}"]))
        grupo += 1
        restantes -= k
    # parados sozinhos
    for _ in range(n_parado):
        p = _ponto_dentro(rng, area_in)
        yaw = rng.uniform(-180, 180)
        pessoas.append(Pessoa(novo_id(), "parado", -1, 0.0, [(0.0, p[0], p[1], yaw), (duracao_s, p[0], p[1], yaw)],
                              [f"LookAround {duracao_s:.1f}" if rng.random() < 0.5 else f"Idle {duracao_s:.1f}"]))
    # grupos andando (lider + deslocamentos fixos no mundo)
    restantes = n_gandando
    while restantes > 0:
        k = int(min(restantes, rng.integers(2, 5)))
        vel = float(np.clip(rng.normal(1.1, 0.1), 0.8, 1.4))
        lider = _caminhar_livre(rng, area.buffer(-margem_m - 2), _ponto_dentro(rng, area.buffer(-margem_m - 2)), vel, duracao_s, 0.35)
        offs = [np.zeros(2)]
        while len(offs) < k:
            o = rng.uniform(-1.6, 1.6, 2)
            if all(np.linalg.norm(o - q) > 0.75 for q in offs):
                offs.append(o)
        for o in offs:
            kf = [(t, x + o[0], y + o[1], w) for (t, x, y, w) in lider.kf]
            cmd = []
            for cm in lider.cmd:
                if cm.startswith("GoTo"):
                    _, x, y, z, r = cm.split()
                    cmd.append(f"GoTo {float(x) + o[0]:.2f} {float(y) + o[1]:.2f} 0.0 _")
                else:
                    cmd.append(cm)
            pessoas.append(Pessoa(novo_id(), "grupo_andando", grupo, vel, kf, cmd))
        grupo += 1
        restantes -= k
    # cruzam a fronteira entre subareas (vai-e-volta perpendicular a fronteira)
    for _ in range(n_fronteira):
        for _t in range(200):
            b = fronteiras.interpolate(rng.uniform(0, fronteiras.length))
            b2 = fronteiras.interpolate(min(fronteiras.length, fronteiras.project(b) + 1.0))
            d = np.array([b2.x - b.x, b2.y - b.y])
            if np.linalg.norm(d) < 1e-6:
                continue
            d /= np.linalg.norm(d)
            nrm = np.array([-d[1], d[0]])
            bb = np.array([b.x, b.y])
            amp = rng.uniform(10, 25)
            p0, p1 = bb - amp * nrm, bb + amp * nrm
            if area_in.contains(LineString([tuple(p0), tuple(p1)])):
                break
        else:
            continue
        vel = float(np.clip(rng.normal(1.3, 0.12), 0.9, 1.7))
        c = _Caminhada(p0, 0.0, vel)
        lado = 1
        while c.t < duracao_s:
            q = (p1 if lado > 0 else p0) + d * rng.uniform(-6, 6)
            if area_in.contains(Point(q)):
                c.ir(q)
            lado *= -1
            if rng.random() < 0.3:
                c.parar(float(rng.uniform(2, 10)))
        pessoas.append(Pessoa(novo_id(), "cruza_fronteira", -1, vel, c.kf, c.cmd))
    # caminhantes
    while len(pessoas) < n:
        vel = float(np.clip(rng.normal(1.3, 0.15), 0.9, 1.7))
        if linhas and rng.random() < 0.5:
            c = _caminhar_no_caminho(rng, area_in, linhas[int(rng.integers(len(linhas)))], vel, duracao_s)
        else:
            c = _caminhar_livre(rng, area_in, _ponto_dentro(rng, area_in), vel, duracao_s)
        pessoas.append(Pessoa(novo_id(), "caminhante", -1, vel, c.kf, c.cmd))
    return pessoas[:n]


def gabarito(pessoas: list[Pessoa], duracao_s: float, hz: float = 10.0):
    """Tabela (t, id, x, y, yaw, andando) amostrada a `hz`."""
    tt = np.arange(0.0, duracao_s + 1e-9, 1.0 / hz)
    linhas = []
    for p in pessoas:
        x, y, w = p.posicao(tt)
        v = np.r_[0.0, np.hypot(np.diff(x), np.diff(y)) * hz]
        for i in range(len(tt)):
            linhas.append((round(tt[i], 3), p.id, round(x[i], 3), round(y[i], 3), round(w[i], 2), int(v[i] > 0.2)))
    return linhas


def comandos_anim_people(pessoas: list[Pessoa]) -> str:
    """Arquivo de comandos no formato do omni.anim.people ("<Personagem> <Comando> <args>")."""
    out = []
    for p in pessoas:
        for c in p.comandos:
            out.append(f"{p.id} {c}")
    return "\n".join(out) + "\n"
