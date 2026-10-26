"""Plano B: envia e executa as missoes .plan (exportadas pelo cerise_drones) nas N instancias PX4 SITL via MAVSDK.

Cada instancia i escuta MAVLink em udp://:14540+i (padrao do PX4 SITL multi-veiculo) e ganha um
mavsdk_server proprio (porta gRPC 50050+i). As missoes comecam juntas (asyncio.gather).
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from mavsdk import System

PARAMS_SITL = {"COM_RC_IN_MODE": 4, "NAV_RCL_ACT": 0, "NAV_DLL_ACT": 0, "COM_RCL_EXCEPT": 4}


async def voar(inst, cen_dir: Path, t_lim: float):
    i = inst["instancia_px4"]
    d = System(port=50050 + i)
    await d.connect(system_address=f"udp://:{inst['porta_mavsdk']}")
    async for s in d.core.connection_state():
        if s.is_connected:
            break
    print(f"[{inst['drone']}] conectado", flush=True)
    for k, v in PARAMS_SITL.items():
        try:
            await d.param.set_param_int(k, v)
        except Exception as e:  # noqa: BLE001
            print(f"[{inst['drone']}] param {k}: {e!r}", flush=True)
    async for h in d.telemetry.health():
        if h.is_global_position_ok and h.is_home_position_ok:
            break
    plan = cen_dir.parent / inst["plan"]
    dados = await d.mission_raw.import_qgroundcontrol_mission(str(plan))
    await d.mission_raw.upload_mission(dados.mission_items)
    print(f"[{inst['drone']}] missao enviada ({len(dados.mission_items)} itens) de {plan.name}", flush=True)
    await d.action.arm()
    await d.mission_raw.start_mission()
    t0 = time.time()
    async for pr in d.mission_raw.mission_progress():
        print(f"[{inst['drone']}] {pr.current}/{pr.total} ({time.time() - t0:.0f} s)", flush=True)
        if pr.current >= pr.total or time.time() - t0 > t_lim:
            break
    async for armado in d.telemetry.armed():
        if not armado or time.time() - t0 > t_lim:
            break
    print(f"[{inst['drone']}] fim", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instancias", required=True)
    ap.add_argument("--cenario", required=True)
    ap.add_argument("--t_lim", type=float, default=1800.0, help="limite de relogio por missao (s)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.instancias).read_text())
    await asyncio.gather(*(voar(i, Path(a.cenario), a.t_lim) for i in cfg["instancias"]))


if __name__ == "__main__":
    asyncio.run(main())
