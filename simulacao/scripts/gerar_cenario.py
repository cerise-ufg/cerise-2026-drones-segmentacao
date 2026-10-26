"""Gera o cenario da simulacao (independente do simulador) a partir do plano da UFG da frente de drones.

Saida em <saida>/<nome>/:
  cenario.json            origem ENU, area e subareas (ENU), drones (parametros, camera, trajetoria
                          temporizada, rota), camera de visao geral, duracao, resumo
  cena.json               layout procedural (malhas do terreno, caminhos, edificios, arvores)
  pessoas_plano.json      plano das pessoas (quadros-chave + comandos omni.anim.people)
  pessoas_comandos.txt    comandos no formato do omni.anim.people
  gabarito_pessoas_plano.csv   verdade-terreno do PLANO (t,id,x,y,yaw,andando) a --hz_gabarito
  drones/<id>_trajetoria.csv   t,x,y,z,yaw,cobre,fase a --hz_gabarito
  drones/<id>_camera.csv       t,cx,cy,cz,qw,qx,qy,qz (R_wc OpenCV->ENU),q_usd(w,x,y,z)
  missoes/                .plan (QGC/PX4), .kml, .geojson em lon/lat (exportador do cerise_drones)
  texturas/*.jpg          texturas procedurais ladrilhaveis
  mapa_cenario.png        figura do cenario
  sem_render/             execucao sintetica sem render (poses + caixas projetadas) p/ testes
Uso:
  python scripts/gerar_cenario.py --nome reduzido --recorte -300,-90,110,360 --n_pessoas 80
  python scripts/gerar_cenario.py --nome completo --n_pessoas 120
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SIM = AQUI.parent
RAIZ = SIM.parent
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import box, mapping  # noqa: E402

from cerise_drones.cenarios import area_ufg, bases_no_contorno  # noqa: E402
from cerise_drones.frota import criar_frota  # noqa: E402
from cerise_drones.geo import AreaMissao, poligonos  # noqa: E402
from cerise_drones.planejador import planejar_missao  # noqa: E402
from cerise_sim import texturas  # noqa: E402
from cerise_sim.camera import (Intrinsecos, olhar_para, pose_drone, quat_de_matriz,  # noqa: E402
                               r_usd_de_cv)
from cerise_sim.cena import gerar_layout  # noqa: E402
from cerise_sim.dados import escrever_csv  # noqa: E402
from cerise_sim.enu import ENU  # noqa: E402
from cerise_sim.pessoas import comandos_anim_people, gabarito, gerar_pessoas  # noqa: E402
from cerise_sim.sintetico import simular_sem_render  # noqa: E402
from cerise_sim.trajetoria import construir  # noqa: E402

ASPECTOS = {"4:3": 4 / 3, "16:9": 16 / 9, "3:2": 3 / 2}


def resolucao_para_fov(hfov, vfov, largura):
    asp = math.tan(math.radians(hfov) / 2) / math.tan(math.radians(vfov) / 2)
    nome, a = min(ASPECTOS.items(), key=lambda kv: abs(kv[1] - asp))
    altura = int(round(largura / a / 8) * 8)
    return nome, largura, altura


def coords_poly(g):
    return [[list(map(lambda v: round(float(v), 3), c)) for c in p.exterior.coords] for p in poligonos(g)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", default="reduzido")
    ap.add_argument("--recorte", default=None,
                    help="xmin,ymin,xmax,ymax em metros relativos ao centroide da area UFG (UTM); vazio = area completa")
    ap.add_argument("--altitude", type=float, default=40.0)
    ap.add_argument("--n_pessoas", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--largura", type=int, default=1280, help="largura das imagens dos drones (px)")
    ap.add_argument("--inclinacao", type=float, default=0.0, help="graus a partir do nadir (0 = nadir)")
    ap.add_argument("--hz_gabarito", type=float, default=10.0)
    ap.add_argument("--fps_sem_render", type=float, default=5.0)
    ap.add_argument("--saida", default=str(SIM / "cenario"))
    ap.add_argument("--sem_texturas", action="store_true")
    a = ap.parse_args()
    t_ini = time.time()
    out = Path(a.saida) / a.nome
    (out / "drones").mkdir(parents=True, exist_ok=True)

    cheia = area_ufg()
    if a.recorte:
        x0, y0, x1, y1 = map(float, a.recorte.split(","))
        c = cheia.livre.centroid
        sub = cheia.livre.intersection(box(c.x + x0, c.y + y0, c.x + x1, c.y + y1))
        sub = max(poligonos(sub), key=lambda p: p.area)
        sub = shapely.simplify(sub, 0.5)
        area = AreaMissao(f"UFG Samambaia - recorte {a.nome}", cheia.proj.lonlat(sub), [], 10.0, projetor=cheia.proj)
    else:
        area = cheia
    frota = criar_frota(["A", "B", "C", "A"], bases_no_contorno(area, 4), altitude_m=a.altitude)
    plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
    m = plano.metricas
    print(f"area {area.livre.area / 1e4:.2f} ha; makespan {m['makespan_s'] / 60:.2f} min; CV {m['cv_tempo']:.3f}; "
          f"cobertura {m['cobertura_pct']:.2f}%")

    c_ll = area.area_ll.centroid
    enu = ENU(round(c_ll.x, 6), round(c_ll.y, 6), area.proj.epsg)
    area_enu = shapely.make_valid(enu.geom_de_utm(area.livre))
    regioes_enu = [shapely.make_valid(enu.geom_de_utm(r)) for r in plano.particao.regioes]

    drones = []
    duracao = 0.0
    for d, r, mi in zip(frota, plano.rotas, plano.missoes):
        ex, ey = enu.de_utm(r.pontos[:, 0], r.pontos[:, 1])
        tr = construir(d.id, np.column_stack([ex, ey]), r.cobre, d.altitude_m, d.velocidade_ms, d.vel_subida_ms,
                       d.vel_descida_ms, d.tempo_curva_s)
        duracao = max(duracao, tr.duracao_s)
        asp, W, H = resolucao_para_fov(d.hfov_graus, d.vfov_graus, a.largura)
        K = Intrinsecos.de_hfov(W, H, d.hfov_graus)
        bx, by = enu.de_utm(*d.partida_xy)
        lon, lat = area.proj.ponto_lonlat(*d.partida_xy)
        drones.append({
            "id": d.id, "tipo": d.tipo, "rotulo": d.rotulo, "velocidade_ms": d.velocidade_ms,
            "altitude_m": d.altitude_m, "hfov_graus_plano": d.hfov_graus, "vfov_graus_plano": d.vfov_graus,
            "base_enu": [round(float(bx), 3), round(float(by), 3)], "base_lonlat": [lon, lat],
            "camera": {"intrinsecos": K.para_dict(), "aspecto": asp, "inclinacao_graus": a.inclinacao,
                       "montagem": "abaixo do corpo, eixo optico para baixo; topo da imagem = direcao de voo",
                       "gsd_cm_px_nadir": round(100 * d.altitude_m / K.fx, 2),
                       "pessoa_0_5m_em_px": round(0.5 * K.fx / d.altitude_m, 1)},
            "trajetoria": tr.para_dict(),
            "tempo_missao_s_modelo_cerise_drones": round(float(mi.tempo_total_s), 2),
            "tempo_missao_s_trajetoria": round(tr.duracao_s, 2),
            "distancia_m": round(r.distancia_m, 1), "curvas": int(r.n_curvas),
            "area_subarea_m2": round(float(regioes_enu[len(drones)].area), 1),
            "plan_qgc": f"missoes/{a.nome}_{d.id}.plan",
        })
        # CSVs amostrados
        tt = np.arange(0, tr.duracao_s + 1e-9, 1.0 / a.hz_gabarito)
        x, y, z, yaw, cobre, fase = tr.amostrar(tt)
        escrever_csv(out / "drones" / f"{d.id}_trajetoria.csv", ["t", "x", "y", "z", "yaw", "cobre", "fase"],
                     [(round(t, 3), round(x[i], 3), round(y[i], 3), round(z[i], 3), round(yaw[i], 3), int(cobre[i]), fase[i])
                      for i, t in enumerate(tt)])
        linhas = []
        for i, t in enumerate(tt):
            ps = pose_drone(x[i], y[i], z[i], yaw[i], a.inclinacao)
            q = quat_de_matriz(ps.R_wc)
            qu = quat_de_matriz(r_usd_de_cv(ps.R_wc))
            linhas.append((round(t, 3), *np.round(ps.C, 3), *np.round(q, 6), *np.round(qu, 6)))
        escrever_csv(out / "drones" / f"{d.id}_camera.csv",
                     ["t", "cx", "cy", "cz", "qw", "qx", "qy", "qz", "usd_qw", "usd_qx", "usd_qy", "usd_qz"], linhas)
    duracao = float(math.ceil(duracao + 2.0))

    # camera de visao geral: alta, a sudoeste da area, olhando para o centroide
    minx, miny, maxx, maxy = area_enu.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    cen = area_enu.centroid
    pos_vg = [cen.x - 0.55 * (maxx - minx), cen.y - 0.75 * (maxy - miny), 0.55 * diag]
    alvo_vg = [cen.x, cen.y + 0.05 * (maxy - miny), 0.0]
    Kvg = Intrinsecos.de_hfov(1920, 1080, 62.0)
    Rvg = olhar_para(pos_vg, alvo_vg)
    visao_geral = {"posicao": [round(v, 3) for v in pos_vg], "alvo": [round(v, 3) for v in alvo_vg],
                   "intrinsecos": Kvg.para_dict(), "q_cv": quat_de_matriz(Rvg).round(6).tolist(),
                   "q_usd": quat_de_matriz(r_usd_de_cv(Rvg)).round(6).tolist()}

    layout = gerar_layout(area_enu, seed=a.seed)
    fronteiras = shapely.union_all([g.boundary for g in regioes_enu]).difference(area_enu.boundary.buffer(1.0))
    pessoas = gerar_pessoas(area_enu, a.n_pessoas, duracao, seed=a.seed, caminhos=layout["caminhos"], fronteiras=fronteiras)
    tipos = {}
    for p in pessoas:
        tipos[p.tipo] = tipos.get(p.tipo, 0) + 1
    print("pessoas:", tipos)
    escrever_csv(out / "gabarito_pessoas_plano.csv", ["t", "pessoa_id", "x", "y", "yaw", "andando"],
                 gabarito(pessoas, duracao, a.hz_gabarito))
    (out / "pessoas_plano.json").write_text(json.dumps({"duracao_s": duracao, "pessoas": [p.para_dict() for p in pessoas]}),
                                            encoding="utf-8")
    (out / "pessoas_comandos.txt").write_text(comandos_anim_people(pessoas), encoding="utf-8")
    (out / "cena.json").write_text(json.dumps(layout), encoding="utf-8")
    tex = {} if a.sem_texturas else texturas.salvar_todas(out / "texturas", seed=a.seed)
    arqs = plano.exportar(out / "missoes", a.nome)

    cenario = {
        "nome": a.nome, "versao": 1, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "origem_enu": enu.para_dict(),
        "area_enu": coords_poly(area_enu), "area_ha": round(area_enu.area / 1e4, 3),
        "regioes_enu": {d["id"]: coords_poly(g) for d, g in zip(drones, regioes_enu)},
        "fronteiras_enu": mapping(fronteiras) if not fronteiras.is_empty else None,
        "drones": drones, "visao_geral": visao_geral,
        "duracao_s": duracao, "texturas": tex,
        "arquivos": {"cena": "cena.json", "pessoas": "pessoas_plano.json", "comandos_anim_people": "pessoas_comandos.txt",
                     "gabarito_plano": "gabarito_pessoas_plano.csv"},
        "plano_cerise_drones": {"metodo": m["metodo"], "makespan_s": m["makespan_s"], "cv_tempo": m["cv_tempo"],
                                "cobertura_pct": m["cobertura_pct"], "sobreposicao_lateral": plano.sobreposicao_lateral,
                                "sobreposicao_frontal": plano.sobreposicao_frontal},
        "pessoas_por_tipo": tipos, "n_pessoas": len(pessoas),
        "parametros": vars(a),
    }
    (out / "cenario.json").write_text(json.dumps(cenario, indent=1), encoding="utf-8")

    # execucao sintetica sem render (para testes do pipeline e estimativas)
    sr = simular_sem_render(cenario, pessoas, a.fps_sem_render, out / "sem_render")
    alturas = []
    for cam, cx in sr["caixas"].items():
        for (_k, _pid, x1, y1, x2, y2, _o) in cx:
            alturas.append(max(x2 - x1, y2 - y1))
    resumo = {
        "area_ha": cenario["area_ha"], "n_pessoas": len(pessoas), "pessoas_por_tipo": tipos, "duracao_s": duracao,
        "makespan_plano_s": round(m["makespan_s"], 2),
        "tempos_trajetoria_s": {d["id"]: d["tempo_missao_s_trajetoria"] for d in drones},
        "tempos_modelo_cerise_drones_s": {d["id"]: d["tempo_missao_s_modelo_cerise_drones"] for d in drones},
        "resolucoes": {d["id"]: f'{d["camera"]["intrinsecos"]["largura"]}x{d["camera"]["intrinsecos"]["altura"]}' for d in drones},
        "gsd_cm_px": {d["id"]: d["camera"]["gsd_cm_px_nadir"] for d in drones},
        "quadros_por_camera": {f"{f}fps": int(duracao * f) + 1 for f in (2, 5, 10)},
        "caixas_projetadas_sem_render": len(alturas),
        "lado_maior_caixa_px": {"mediana": float(np.median(alturas)) if alturas else None,
                                "p10": float(np.percentile(alturas, 10)) if alturas else None,
                                "p90": float(np.percentile(alturas, 90)) if alturas else None},
        "edificios": len(layout["edificios"]), "arvores": len(layout["arvores"]), "caminhos": len(layout["caminhos"]),
        "tempo_geracao_s": round(time.time() - t_ini, 1),
    }
    (out / "resumo.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")
    print(json.dumps(resumo, indent=1))
    try:
        from figura_cenario import figura
        figura(out, out / "mapa_cenario.png")
    except Exception as e:
        print("figura falhou:", repr(e))
    print("ok ->", out, [x.name for x in arqs][:2], "...")


if __name__ == "__main__":
    main()
