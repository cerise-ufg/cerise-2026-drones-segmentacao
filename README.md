# cerise-2026-drones-segmentacao

Software de **segmentação da área de atuação entre os drones disponíveis** (Atividade 13 do Plano de
Trabalho CERISE/EMC/UFG 2026), com planejamento de cobertura por drone, **colaboração** (realocação
dinâmica quando um drone falha ou fica com bateria baixa), conjunto de **mensagens LoRa** compatível com
a frente LoRa e **pipeline de detecção e contagem de pessoas** em imagens aéreas (YOLO11 no VisDrone,
treinado/avaliado no DGX Spark).

> Bolsista: Eduardo Martins de Carvalho Caixeta - CERISE/EMC/UFG - 2026.

## O que tem aqui

| Pasta | Conteúdo |
|---|---|
| `cerise_drones/` | pacote Python: `geo` (GeoJSON lon/lat -> UTM), `frota` (drone, câmera, bateria), `particao` (faixas, k-means/Voronoi, DARP em grade ponderado), `cobertura` (boustrophedon, orientação ótima, trânsito evitando zonas proibidas, bateria com trocas), `metricas`, `exportar` (QGroundControl `.plan`, KML, GeoJSON), `visualizar` (PNG e mapa folium), `colaboracao` (falha + repartição/leilão), `protocolo` (quadros LoRa `cerise_proto` v1 + extensão), `agregacao` (georreferenciamento e deduplicação de contagens) |
| `dados/` | polígonos do Campus Samambaia/UFG extraídos do OpenStreetMap (ODbL) |
| `scripts/` | `demo_ufg.py`, `benchmark.py`, `graficos_benchmark.py`, `colaboracao_sim.py`, `protocolo_analise.py`, `agregacao_sim.py`, `figura_cenarios.py` |
| `deteccao/` | Dockerfile (`FROM nvcr.io/nvidia/pytorch:26.05-py3` + ultralytics), download do VisDrone2019-DET, preparação do subconjunto "pessoa", avaliação do YOLO11 COCO, fine-tuning, avaliação de mAP/contagem/FPS |
| `tests/` | 16 testes (`python -m pytest -q tests`) |
| `resultados/` | métricas medidas (`.csv/.json/.md`), missão exportada da UFG (`missao_ufg/`), métricas de detecção (`deteccao/`) |
| `evidencias/` | figuras usadas no relatório |
| `RELATORIO_SECAO.md` | seção do relatório técnico |

## Como rodar (planejamento)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install numpy shapely pyproj scikit-learn scipy matplotlib folium imageio pillow playwright pandas pytest
.\.venv\Scripts\python -m playwright install chromium     # só para capturar o mapa HTML
.\.venv\Scripts\python scripts\demo_ufg.py                # UFG: 4 drones, exporta .plan/.kml/.geojson + mapa
.\.venv\Scripts\python scripts\benchmark.py               # 4 cenários x N=2..6 x 2 frotas x 4 métodos
.\.venv\Scripts\python scripts\benchmark.py --ablacao
.\.venv\Scripts\python scripts\graficos_benchmark.py
.\.venv\Scripts\python scripts\colaboracao_sim.py         # falha/bateria baixa + realocação (GIF)
.\.venv\Scripts\python -m pytest -q tests
```

Uso como biblioteca:

```python
from cerise_drones import AreaMissao, Drone, planejar_missao
area = AreaMissao.de_arquivos("minha area", "area.geojson", ["zona_proibida.geojson"], margem_zona_m=10)
frota = [Drone("D1", velocidade_ms=10, autonomia_min=20, hfov_graus=70, vfov_graus=55, altitude_m=40), ...]
for d, (lon, lat) in zip(frota, bases): d.partida_xy = area.proj.ponto_xy(lon, lat)
plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
plano.exportar("saida", "missao")      # missao_D1.plan (QGroundControl), .kml, .geojson
```

## Como rodar (detecção, no DGX Spark)

```bash
cd ~/cerise2026/drones && bash deteccao/build.sh               # imagem cerise26_drones:latest
NP=12 bash deteccao/00_fetch_visdrone_parallel.sh              # zips oficiais (github.com/ultralytics/assets)
bash deteccao/run.sh prep 02_prep_pessoa.py                     # pedestrian+people -> "pessoa"
bash deteccao/run.sh treino 04_treinar.py --modelo yolo11s.pt --imgsz 960 --batch 4 --horas 0.8
bash deteccao/run.sh avaliacao 06_avaliar_tudo.py               # mAP COCO x ajustado, contagem, FPS
```

Observação: no GB10 (PyTorch 2.12 NGC 26.05) o autocast FP16 produziu NaN (`deteccao/teste_amp.py`);
treino e inferência foram feitos em FP32 (`amp=False`, `half=False`).

## Resultados (resumo)

Ver `RELATORIO_SECAO.md` para as tabelas completas. Todos os números saem de `resultados/`.

Não versionados: o dataset VisDrone (`~/cerise2026/drones/datasets`, ~2 GB), os pesos treinados
(`~/cerise2026/drones/runs/ft_pessoa_yolo11s_960/weights/best.pt`, 19 MB, no Spark) e o `.venv`.
