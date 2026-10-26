"""Mensagens LoRa entre drones e estacao de solo - implementacao Python compativel, byte a byte, com a
biblioteca `cerise_proto` v1 da frente LoRa (firmware/lib/cerise_proto, C99), mais uma PROPOSTA de
extensao (tipos 7-9) necessaria para a colaboracao (poligonos arbitrarios e leilao de tarefas).

Quadro (little-endian):  ver_type(u8 = versao<<4 | tipo) | flags(u8) | net_id(u8) | src(u8) |
                         dst(u8, 0xFF broadcast, 0x00 gateway) | seq(u16) | len(u8) | payload | crc16
CRC-16/CCITT-FALSE (poli 0x1021, init 0xFFFF) sobre cabecalho+payload.

Tipos v1 (frente LoRa):  1 HEARTBEAT (8 B) | 2 POS (14 B) | 3 AREA_ASSIGN (22 B, retangulo envolvente)
                         4 COVERAGE_DONE (7 B) | 5 COUNT_REPORT (13 B) | 6 ACK (6 B)
Extensao proposta:       7 AREA_POLY (4 + 4n B, n <= 7 vertices por quadro; fragmentado)
                         8 TASK_BID (4 B) | 9 TASK_AWARD (3 B)
"""
from __future__ import annotations

import math
import struct

VERSAO = 1
CAB = struct.Struct("<BBBBBHB")          # 8 bytes
TAM_CRC = 2
MAX_PAYLOAD = 32
GATEWAY, BROADCAST = 0x00, 0xFF

HEARTBEAT, POS, AREA_ASSIGN, COVERAGE_DONE, COUNT_REPORT, ACK, AREA_POLY, TASK_BID, TASK_AWARD = range(1, 10)
NOMES = {HEARTBEAT: "HEARTBEAT", POS: "POS", AREA_ASSIGN: "AREA_ASSIGN", COVERAGE_DONE: "COVERAGE_DONE",
         COUNT_REPORT: "COUNT_REPORT", ACK: "ACK", AREA_POLY: "AREA_POLY", TASK_BID: "TASK_BID",
         TASK_AWARD: "TASK_AWARD"}
# estados do HEARTBEAT (cp_state_t)
EST_IDLE, EST_ARMED, EST_MISSION, EST_RTL, EST_LANDED, EST_FAILSAFE = range(6)
FLAG_ACK_REQ, FLAG_RETRY = 0x01, 0x02

PAYLOAD = {
    HEARTBEAT: struct.Struct("<BHBI"),        # batt_pct, batt_mv, state, uptime_s
    POS: struct.Struct("<iihBHB"),            # lat_e7, lon_e7, alt_dm, batt_pct, people, area_id
    AREA_ASSIGN: struct.Struct("<BBiiiihH"),  # area_id, task_id, lat_min, lon_min, lat_max, lon_max, alt_dm, spacing_dm
    COVERAGE_DONE: struct.Struct("<BBBHH"),   # area_id, task_id, coverage_pct, duration_s, people_total
    COUNT_REPORT: struct.Struct("<BBHBii"),   # area_id, task_id, people, confidence_pct, lat_e7, lon_e7
    ACK: struct.Struct("<HBBBb"),             # acked_seq, acked_type, status, rssi_neg, snr_qdb
    AREA_POLY: struct.Struct("<BBBB"),        # area_id, task_id, frag (idx<<4 | total), n  (+ n x (dx,dy) i16 m)
    TASK_BID: struct.Struct("<BBH"),          # task_id, area_id, custo_ds (tempo de conclusao estimado, 0,1 s)
    TASK_AWARD: struct.Struct("<BBB"),        # task_id, area_id, vencedor
}
VERT = struct.Struct("<hh")
assert [PAYLOAD[t].size for t in range(1, 7)] == [8, 14, 22, 7, 13, 6]  # = cp_payload_len() da frente LoRa


def crc16(dados: bytes) -> int:
    crc = 0xFFFF
    for b in dados:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def codificar(tipo, campos, src, dst=GATEWAY, seq=0, net_id=1, flags=0, extra: bytes = b"") -> bytes:
    payload = PAYLOAD[tipo].pack(*campos) + extra
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} B > {MAX_PAYLOAD} B")
    q = CAB.pack((VERSAO << 4) | tipo, flags, net_id, src, dst, seq & 0xFFFF, len(payload)) + payload
    return q + struct.pack("<H", crc16(q))


def decodificar(q: bytes) -> dict:
    if len(q) < CAB.size + TAM_CRC:
        raise ValueError("curto")
    if crc16(q[:-2]) != struct.unpack("<H", q[-2:])[0]:
        raise ValueError("CRC")
    vt, flags, net, src, dst, seq, n = CAB.unpack_from(q, 0)
    if vt >> 4 != VERSAO:
        raise ValueError("versao")
    tipo = vt & 0x0F
    st = PAYLOAD[tipo]
    campos = st.unpack_from(q, CAB.size)
    extra = q[CAB.size + st.size:CAB.size + n]
    out = {"tipo": NOMES[tipo], "flags": flags, "net_id": net, "src": src, "dst": dst, "seq": seq, "campos": campos}
    if tipo == AREA_POLY:
        out["vertices"] = [VERT.unpack_from(extra, 4 * i) for i in range(campos[3])]
    return out


# ------------------------------------------------------------------------------------ construtores
def e7(v):
    return int(round(v * 1e7))


def area_assign_poligono(area_id, task_id, poligono_ll, alt_m, espacamento_m, src=GATEWAY, dst=BROADCAST,
                         seq=0, proj=None, tol_m=2.0) -> list[bytes]:
    """AREA_ASSIGN (retangulo envolvente, v1) + fragmentos AREA_POLY com os vertices do poligono
    (offsets em metros a partir do canto (lon_min, lat_min)). O poligono e simplificado (tol_m)."""
    from shapely.geometry import Polygon
    lon0, lat0, lon1, lat1 = poligono_ll.bounds
    quadros = [codificar(AREA_ASSIGN, (area_id, task_id, e7(lat0), e7(lon0), e7(lat1), e7(lon1), int(round(alt_m * 10)),
                                       int(round(espacamento_m * 10))), src, dst, seq)]
    if proj is None:
        return quadros
    pxy = proj.xy(poligono_ll)
    if tol_m:
        pxy = pxy.simplify(tol_m, preserve_topology=True)
    ref = proj.ponto_xy(lon0, lat0)
    partes = list(getattr(pxy, "geoms", [pxy]))
    verts = []
    for p in partes:  # partes separadas por um vertice sentinela (-32768, -32768)
        if verts:
            verts.append((-32768, -32768))
        verts += [(int(round(x - ref[0])), int(round(y - ref[1]))) for x, y in list(p.exterior.coords)[:-1]]
    por = (MAX_PAYLOAD - PAYLOAD[AREA_POLY].size) // VERT.size  # 7
    frags = [verts[i:i + por] for i in range(0, len(verts), por)]
    for k, fr in enumerate(frags):
        extra = b"".join(VERT.pack(*v) for v in fr)
        quadros.append(codificar(AREA_POLY, (area_id, task_id, (k << 4) | len(frags), len(fr)), src, dst, seq + 1 + k,
                                 extra=extra))
    return quadros


def tempo_no_ar_ms(n_bytes: int, sf: int = 7, bw_hz: float = 125e3, cr_denom: int = 5, preambulo: int = 8,
                   cabecalho_explicito: bool = True, crc: bool = True) -> float:
    """Semtech AN1200.13 (mesma formula de cp_airtime_us da frente LoRa)."""
    t_sym = (2 ** sf) / bw_hz
    de = 1 if t_sym >= 0.016 else 0
    ih = 0 if cabecalho_explicito else 1
    num = 8 * n_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * cr_denom, 0)
    return ((preambulo + 4.25) + n_payload) * t_sym * 1000.0


def exemplos() -> dict:
    """Quadro tipico de cada mensagem (para a tabela de tamanhos/tempo no ar)."""
    return {
        "HEARTBEAT": codificar(HEARTBEAT, (87, 15800, EST_MISSION, 312), 1),
        "POS": codificar(POS, (e7(-16.604), e7(-49.265), 400, 87, 12, 3), 1),
        "AREA_ASSIGN": codificar(AREA_ASSIGN, (3, 1, e7(-16.61), e7(-49.28), e7(-16.60), e7(-49.26), 400, 448), 0, 1),
        "AREA_POLY (7 vert.)": codificar(AREA_POLY, (3, 1, 0x02, 7), 0, 1, extra=b"".join(VERT.pack(i * 10, i * 3) for i in range(7))),
        "COVERAGE_DONE": codificar(COVERAGE_DONE, (3, 1, 100, 412, 57), 1),
        "COUNT_REPORT": codificar(COUNT_REPORT, (3, 1, 57, 71, e7(-16.604), e7(-49.265)), 1),
        "ACK": codificar(ACK, (17, POS, 0, 92, 30), 0, 1),
        "TASK_BID": codificar(TASK_BID, (5, 9, 4200), 2, 0),
        "TASK_AWARD": codificar(TASK_AWARD, (5, 9, 2), 0, BROADCAST),
    }
