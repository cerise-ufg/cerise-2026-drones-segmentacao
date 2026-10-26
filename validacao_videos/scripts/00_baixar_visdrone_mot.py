"""Download resumivel do VisDrone2019-MOT (val e test-dev) a partir dos links OFICIAIS do Google Drive
listados em https://github.com/VisDrone/VisDrone-Dataset (README, secao "VisDrone-MOT dataset").

O Drive devolve intermitentemente a pagina "Quota exceeded" (arquivo popular). O script pede o intervalo
[bytes ja baixados .. fim] numa unica conexao; se vier HTML em vez do zip, espera e tenta de novo,
retomando de onde parou. So usa a biblioteca padrao (roda no host do Spark com python3).
Uso: python3 00_baixar_visdrone_mot.py [destino]
"""
import hashlib
import os
import sys
import time
import urllib.request
import zipfile

DEST = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/cerise2026/drones/mot")
ARQS = [("VisDrone2019-MOT-val.zip", "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"),
        ("VisDrone2019-MOT-test-dev.zip", "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu")]
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
CHUNK = 1 << 20
os.makedirs(f"{DEST}/zips", exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tamanho_total(fid):
    for _ in range(1000):
        req = urllib.request.Request(URL.format(fid), headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if r.status == 206 and cr:
                    return int(cr.split("/")[-1])
        except Exception as e:  # noqa
            log("erro ao obter tamanho:", e)
        time.sleep(30)
    raise SystemExit("nao foi possivel obter o tamanho")


for nome, fid in ARQS:
    final = f"{DEST}/zips/{nome}"
    if os.path.exists(final):
        log(nome, "ja existe"); continue
    parcial = final + ".part"
    total = tamanho_total(fid)
    log(nome, "tamanho", total)
    espera, tent = 30, 0
    while True:
        feito = os.path.getsize(parcial) if os.path.exists(parcial) else 0
        if feito >= total:
            break
        tent += 1
        req = urllib.request.Request(URL.format(fid), headers={"Range": f"bytes={feito}-", "User-Agent": "curl/8"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                ct = r.headers.get("Content-Type", "")
                if r.status != 206 or "html" in ct:
                    log(f"tentativa {tent}: quota/HTML (status {r.status}, {ct}); feito {feito/1e6:.0f} MB; espera {espera}s")
                    time.sleep(espera); espera = min(300, int(espera * 1.5)); continue
                espera = 30
                log(f"tentativa {tent}: baixando a partir de {feito/1e6:.0f} MB")
                t0, n = time.time(), 0
                with open(parcial, "ab") as f:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        f.write(b); n += len(b)
                        if n % (200 * CHUNK) < CHUNK:
                            log(f"  {(feito+n)/1e6:.0f}/{total/1e6:.0f} MB, {n/1e6/(time.time()-t0):.1f} MB/s")
        except Exception as e:  # conexao caiu: retoma
            log(f"tentativa {tent}: erro {e!r}; retomando")
            time.sleep(10)
    os.replace(parcial, final)
    h = hashlib.sha256()
    with open(final, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    log(nome, "OK", os.path.getsize(final), "sha256", h.hexdigest())
    pasta = nome[:-4]
    if not os.path.isdir(f"{DEST}/{pasta}"):
        with zipfile.ZipFile(final) as z:
            bad = z.testzip()
            log("testzip:", bad)
            z.extractall(DEST)
    seqs = sorted(os.listdir(f"{DEST}/{pasta}/sequences"))
    nq = sum(len(os.listdir(f"{DEST}/{pasta}/sequences/{s}")) for s in seqs)
    log(f"{pasta}: {len(seqs)} sequencias, {nq} quadros")
log("FIM_DOWNLOAD")
