"""Video para o relatorio (1920x1080): visao geral da cena (drones identificados por projecao da pose) +
paineis das cameras dos 4 drones com caixas de gabarito (verde) e trilhas do rastreador (vermelho, id do
cluster da contagem unica) + painel com contagem atual (por drone) e acumulada (estimada x gabarito).

Grava os quadros JPEG em --quadros_dir; o MP4 H.264 e montado depois com ffmpeg (ver --ffmpeg).
Uso (conteiner cerise26_drones_sim):
  python scripts/video_relatorio.py --execucao saidas/run01 --contagem resultados/contagem_run01.json \
      --deteccoes resultados/contagem_run01_deteccoes.csv --quadros_dir tmp/video_run01
Depois (host): ffmpeg -framerate 15 -i tmp/video_run01/%06d.jpg -c:v libx264 -crf 26 -pix_fmt yuv420p video.mp4
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cerise_sim.avaliacao import filtrar_gt  # noqa: E402
from cerise_sim.camera import Intrinsecos, projetar  # noqa: E402
from cerise_sim.dados import Execucao  # noqa: E402

W, H = 1920, 1080
CORES = {"D1": (180, 119, 31), "D2": (40, 39, 214), "D3": (44, 160, 44), "D4": (189, 103, 148)}  # BGR
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def encaixar(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 25, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out, s, x0, y0


def texto(img, s, xy, esc=0.55, cor=(255, 255, 255), esp=1, fundo=None):
    if fundo is not None:
        (tw, th), _ = cv2.getTextSize(s, FONTE, esc, esp)
        cv2.rectangle(img, (xy[0] - 3, xy[1] - th - 4), (xy[0] + tw + 3, xy[1] + 4), fundo, -1)
    cv2.putText(img, s, xy, FONTE, esc, cor, esp, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execucao", required=True)
    ap.add_argument("--contagem", required=True, help="json do contar.py (serie temporal do rastreador principal)")
    ap.add_argument("--deteccoes", required=True)
    ap.add_argument("--quadros_dir", required=True)
    ap.add_argument("--max_quadros", type=int, default=None)
    ap.add_argument("--passo", type=int, default=1)
    ap.add_argument("--modo", default="botsort_cal", help="rastreador cuja serie temporal vai no painel")
    ap.add_argument("--rotulo", default="BoT-SORT calibrado (sem fuse_score, limiar 0,15) + dedup no solo")
    ap.add_argument("--titulo", default="CERISE/UFG - simulacao Isaac Sim 5.1: 4 drones contando pessoas (YOLO11s + BoT-SORT + dedup no solo)")
    a = ap.parse_args()
    ex = Execucao(a.execucao)
    cont = json.loads(Path(a.contagem).read_text(encoding="utf-8"))
    modo = a.modo if a.modo in cont else cont["principal"]
    st = cont[modo]["serie_temporal"]
    est_acum = np.array(st["estimada_acumulada"]); gab_acum = np.array(st["vistas_acumulada_gabarito"]); Ts = np.array(st["t"])
    n_cena = cont[modo]["avaliacao_contagem"]["pessoas_na_cena"]
    drones = ex.cameras("drone")[:4]
    quadros = {c: {q.frame: q for q in ex.quadros(c)} for c in ex.cameras(None)}
    gts = {c: ex.caixas_gt(c) for c in drones}
    det = defaultdict(list)
    for r in csv.DictReader(open(a.deteccoes, encoding="utf-8")):
        det[(r["camera"], int(r["frame"]))].append(([float(r[k]) for k in ("x1", "y1", "x2", "y2")], int(r["cluster"])))
    Kv = ex.intrinsecos("visao_geral") if "visao_geral" in quadros else None
    frames = sorted(quadros.get("visao_geral", quadros[drones[0]]))[:: a.passo]
    if a.max_quadros:
        frames = frames[: a.max_quadros]
    pasta = Path(a.quadros_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    rastro = defaultdict(list)
    for n, f in enumerate(frames):
        tela = np.full((H, W, 3), 18, np.uint8)
        t = None
        # ---------------- visao geral (1280x720)
        qv = quadros.get("visao_geral", {}).get(f)
        if qv is not None:
            t = qv.t
            img = cv2.imread(str(qv.imagem)) if qv.imagem else None
            if img is None:
                img = np.full((Kv.altura, Kv.largura, 3), 80, np.uint8)
            cel, s, x0, y0 = encaixar(img, 1280, 720)
            for c in drones:
                q = quadros[c].get(f)
                if q is None:
                    continue
                uv, z = projetar(q.pose.C[None], qv.pose, Kv)
                if z[0] > 0 and np.isfinite(uv).all():
                    p = (int(x0 + uv[0, 0] * s), int(y0 + uv[0, 1] * s))
                    rastro[c].append(p)
                    rastro[c] = rastro[c][-60:]
                    if len(rastro[c]) > 1:
                        cv2.polylines(cel, [np.array(rastro[c], np.int32)], False, CORES[c], 2, cv2.LINE_AA)
                    cv2.circle(cel, p, 7, CORES[c], 2, cv2.LINE_AA)
                    texto(cel, c, (p[0] + 9, p[1] - 6), 0.6, (255, 255, 255), 2, fundo=CORES[c])
            texto(cel, "visao geral (camera fixa a sudoeste da area)", (10, 24), 0.6, (255, 255, 255), 1, fundo=(0, 0, 0))
            tela[0:720, 0:1280] = cel
        # ---------------- drones (4 x 480x360)
        atual = {}
        for i, c in enumerate(drones):
            q = quadros[c].get(f)
            x_off = 480 * i
            if q is None:
                continue
            t = q.t if t is None else t
            K = ex.intrinsecos(c)
            img = cv2.imread(str(q.imagem)) if q.imagem else None
            if img is None:
                img = np.full((K.altura, K.largura, 3), 80, np.uint8)
            cel, s, x0, y0 = encaixar(img, 480, 360)
            ids, bx, oc = gts[c].get(f, ([], np.zeros((0, 4)), np.zeros(0)))
            ids2, bx2 = filtrar_gt(ids, bx, oc)
            for b in bx2:
                cv2.rectangle(cel, (int(x0 + b[0] * s) - 1, int(y0 + b[1] * s) - 1), (int(x0 + b[2] * s) + 1, int(y0 + b[3] * s) + 1), (0, 220, 0), 1)
            dd = det.get((c, f), [])
            for b, cl in dd:
                p1 = (int(x0 + b[0] * s) - 3, int(y0 + b[1] * s) - 3); p2 = (int(x0 + b[2] * s) + 3, int(y0 + b[3] * s) + 3)
                cv2.rectangle(cel, p1, p2, (0, 0, 255), 1)
                if cl >= 0:
                    texto(cel, str(cl), (p1[0], p1[1] - 2), 0.35, (0, 0, 255), 1)
            atual[c] = (len(dd), len(bx2), q.cobre)
            cv2.rectangle(cel, (0, 0), (480, 22), CORES[c], -1)
            texto(cel, f"{c}  z={q.pose.C[2]:.0f} m  trilhas={len(dd)}  gabarito={len(bx2)}" + ("" if q.cobre else "  (transito)"), (6, 16), 0.48)
            tela[720:1080, x_off:x_off + 480] = cel
        # ---------------- painel (640x720)
        pn = np.full((720, 640, 3), 245, np.uint8)
        cv2.rectangle(pn, (0, 0), (640, 30), (60, 60, 60), -1)
        texto(pn, "contagem de pessoas", (12, 21), 0.62, (255, 255, 255), 1)
        tt = t if t is not None else 0.0
        k = int(np.searchsorted(Ts, tt, side="right")) - 1
        k = max(k, 0)
        linhas = [(f"t = {tt:5.1f} s   (quadro {f}, 5 quadros/s)", (20, 20, 20))]
        tot_e = sum(v[0] for v in atual.values()); tot_g = sum(v[1] for v in atual.values())
        linhas.append((f"contagem ATUAL (soma dos 4 drones): {tot_e}   gabarito: {tot_g}", (0, 0, 180)))
        for c in drones:
            if c in atual:
                linhas.append((f"   {c}: trilhas {atual[c][0]:2d}  x  gabarito {atual[c][1]:2d}", CORES[c]))
        linhas.append((f"contagem UNICA acumulada (dedup no solo): {int(est_acum[k])}", (0, 0, 180)))
        linhas.append((f"pessoas distintas ja vistas (gabarito): {int(gab_acum[k])}", (0, 120, 0)))
        linhas.append((f"pessoas na cena: {n_cena}", (60, 60, 60)))
        for j, (s_, cor) in enumerate(linhas):
            texto(pn, s_, (16, 58 + 26 * j), 0.55, cor, 1)
        # grafico acumulado
        gx0, gy0, gw, gh = 60, 345, 550, 280
        cv2.rectangle(pn, (gx0, gy0), (gx0 + gw, gy0 + gh), (200, 200, 200), 1)
        ymax = max(n_cena, int(est_acum.max()) if len(est_acum) else 1, 1) * 1.1
        tmax = max(Ts.max(), 1)

        def pt(tv, yv):
            return (int(gx0 + gw * tv / tmax), int(gy0 + gh - gh * yv / ymax))
        for yv in range(0, int(ymax) + 1, 20):
            cv2.line(pn, pt(0, yv), pt(tmax, yv), (225, 225, 225), 1)
            texto(pn, str(yv), (gx0 - 34, pt(0, yv)[1] + 4), 0.4, (90, 90, 90), 1)
        cv2.line(pn, pt(0, n_cena), pt(tmax, n_cena), (150, 150, 150), 1, cv2.LINE_AA)
        for serie, cor in ((gab_acum, (0, 150, 0)), (est_acum, (0, 0, 200))):
            pts = np.array([pt(Ts[j], serie[j]) for j in range(k + 1)], np.int32)
            if len(pts) > 1:
                cv2.polylines(pn, [pts], False, cor, 2, cv2.LINE_AA)
        cv2.line(pn, pt(tt, 0), pt(tt, ymax / 1.1), (120, 120, 120), 1)
        texto(pn, "acumulado x tempo (s): vermelho = estimado, verde = gabarito", (gx0 - 40, gy0 + gh + 26), 0.45, (40, 40, 40), 1)
        texto(pn, "verde: caixa de gabarito (Replicator); vermelho: trilha + id do cluster", (16, 690), 0.45, (40, 40, 40), 1)
        texto(pn, a.rotulo, (16, 710), 0.45, (40, 40, 40), 1)
        tela[0:720, 1280:1920] = pn
        cv2.rectangle(tela, (0, 690), (1280, 720), (0, 0, 0), -1)
        texto(tela, a.titulo, (10, 711), 0.52, (255, 255, 255), 1)
        cv2.imwrite(str(pasta / f"{n:06d}.jpg"), tela, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if n % 100 == 0:
            print(f"quadro {n}/{len(frames)}", flush=True)
    print(pasta, len(frames), "quadros")


if __name__ == "__main__":
    main()
