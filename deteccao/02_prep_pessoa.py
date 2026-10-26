"""Prepara o subconjunto 'pessoa' do VisDrone2019-DET (pedestrian + people -> classe 0 'pessoa').

- Descompacta os zips oficiais (baixados de github.com/ultralytics/assets, mesmas URLs do VisDrone.yaml)
  em /w/datasets/VisDrone_raw, mantendo as anotacoes originais (necessarias para as regioes ignoradas).
- Conversao igual a do VisDrone.yaml do ultralytics (descarta score==0), mas so com as categorias
  1 (pedestrian) e 2 (people).
- Treino: imagens redimensionadas para lado maior 960 px (reduz o custo de decodificacao JPEG com a
  CPU do Spark ocupada); val/test-dev: imagens originais.
- Salva contagens de referencia por imagem e as regioes ignoradas (categoria 0) para a avaliacao
  de contagem.
"""
import json
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2

RAW = Path("/w/datasets/VisDrone_raw")
OUT = Path("/w/datasets/visdrone_pessoa")
ZIPS = Path("/w/datasets/VisDrone")
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
LADO_TREINO = 960


def descompactar():
    RAW.mkdir(parents=True, exist_ok=True)
    for split, nome in SPLITS.items():
        if not (RAW / nome).exists():
            print("descompactando", nome, flush=True)
            with zipfile.ZipFile(ZIPS / f"{nome}.zip") as z:
                z.extractall(RAW)


def processa(args):
    split, nome, ann = args
    img_src = RAW / nome / "images" / (ann.stem + ".jpg")
    im = cv2.imread(str(img_src))
    h, w = im.shape[:2]
    pessoas, ignoradas = [], []
    n_ped = n_peo = 0
    for linha in ann.read_text().strip().splitlines():
        r = linha.split(",")
        if len(r) < 6:
            continue
        x, y, bw, bh = map(int, r[:4])
        score, cat = r[4].strip(), int(r[5])
        if cat == 0 or score == "0":
            ignoradas.append([x, y, bw, bh])
            continue
        if cat in (1, 2):
            pessoas.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")
            n_ped += cat == 1
            n_peo += cat == 2
    dst_img = OUT / "images" / split / img_src.name
    if split == "train":
        esc = LADO_TREINO / max(h, w)
        if esc < 1:
            im = cv2.resize(im, (round(w * esc), round(h * esc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        os.link(img_src, dst_img) if not dst_img.exists() else None
    (OUT / "labels" / split / (ann.stem + ".txt")).write_text("".join(pessoas))
    return img_src.name, len(pessoas), n_ped, n_peo, ignoradas, (w, h)


def main():
    descompactar()
    resumo = {}
    for split, nome in SPLITS.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        anns = sorted((RAW / nome / "annotations").glob("*.txt"))
        with Pool(6) as p:
            res = p.map(processa, [(split, nome, a) for a in anns], chunksize=16)
        gt = {r[0]: {"pessoas": r[1], "pedestrian": r[2], "people": r[3], "ignoradas": r[4], "wh": r[5]} for r in res}
        json.dump(gt, open(OUT / f"gt_{split}.json", "w"))
        tot = sum(r[1] for r in res)
        resumo[split] = {"imagens": len(res), "pessoas": tot, "pedestrian": sum(r[2] for r in res),
                         "people": sum(r[3] for r in res), "imagens_sem_pessoa": sum(1 for r in res if r[1] == 0),
                         "media_por_imagem": tot / len(res)}
        print(split, resumo[split], flush=True)
    (OUT / "visdrone_pessoa.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: pessoa\n")
    json.dump(resumo, open(OUT / "resumo_dataset.json", "w"), indent=1)
    json.dump(resumo, open("/w/resultados/resumo_dataset.json", "w"), indent=1)


if __name__ == "__main__":
    os.makedirs("/w/resultados", exist_ok=True)
    main()
