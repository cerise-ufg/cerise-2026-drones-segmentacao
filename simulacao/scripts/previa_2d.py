"""Pre-visualizacao 2D (vista de cima) do cenario planejado: drones (pegadas das cameras) percorrendo as
subareas e pessoas se movendo segundo o plano. NAO e render 3D; serve para conferir o cenario antes do
Isaac Sim/Gazebo. Gera GIF e um quadro PNG.
Uso: python scripts/previa_2d.py cenario/reduzido evidencias/previa_cenario_reduzido.gif [passo_s]
"""
import json
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MP  # noqa: E402

from cerise_sim.camera import Intrinsecos, pegada_no_chao, pose_drone  # noqa: E402
from cerise_sim.trajetoria import Trajetoria  # noqa: E402

CORES = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def main():
    pasta = Path(sys.argv[1]); saida = Path(sys.argv[2]); passo = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    cen = json.loads((pasta / "cenario.json").read_text(encoding="utf-8"))
    pes = json.loads((pasta / "pessoas_plano.json").read_text(encoding="utf-8"))["pessoas"]
    trs = [(d, Trajetoria.de_dict(d["trajetoria"]), Intrinsecos.de_dict(d["camera"]["intrinsecos"])) for d in cen["drones"]]
    kfs = [np.asarray(p["kf"], float) for p in pes]
    tt = np.arange(0, cen["duracao_s"], passo)
    a = np.array(cen["area_enu"][0])
    quadros = []
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=90)
    for t in tt:
        ax.clear()
        ax.fill(a[:, 0], a[:, 1], fc="#b5cf8f", ec="k", lw=0.6)
        for i, (d, tr, K) in enumerate(trs):
            for pl in cen["regioes_enu"][d["id"]]:
                ax.add_patch(MP(pl, closed=True, fill=False, ec=CORES[i], lw=0.8, ls="--"))
            x, y, z, yaw, cobre, _ = tr.amostrar(t)
            k = tr.t <= t
            ax.plot(tr.xyz[k, 0], tr.xyz[k, 1], "-", color=CORES[i], lw=0.8, alpha=0.6)
            if z[0] > 1:
                pg = pegada_no_chao(pose_drone(x[0], y[0], z[0], yaw[0]), K)
                ax.add_patch(MP(pg[:, :2], closed=True, fc=CORES[i], alpha=0.25 if cobre[0] else 0.08, ec=CORES[i]))
            ax.plot(x[0], y[0], "^", color=CORES[i], ms=7, mec="k")
        P = np.array([[np.interp(t, k[:, 0], k[:, 1]), np.interp(t, k[:, 0], k[:, 2])] for k in kfs])
        ax.plot(P[:, 0], P[:, 1], "k.", ms=3)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Previa 2D do plano (nao e render) - t = {t:5.1f} s / {cen['duracao_s']:.0f} s\n"
                     f"{len(trs)} drones (pegada da camera), {len(pes)} pessoas", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        quadros.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if abs(t - 60.0) < passo / 2:
            fig.savefig(saida.with_suffix(".png"), dpi=150)
    plt.close(fig)
    from PIL import Image
    ims = [Image.fromarray(q).convert("P", palette=Image.ADAPTIVE, colors=64) for q in quadros]
    ims[0].save(saida, save_all=True, append_images=ims[1:], duration=120, loop=0, optimize=True)
    print(saida, len(ims), "quadros")


if __name__ == "__main__":
    main()
