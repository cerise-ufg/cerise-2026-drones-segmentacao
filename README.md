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

Todos os números saem de arquivos em `resultados/` (ver `RELATORIO_SECAO.md`).

- **Campus Samambaia/UFG (34,2 ha), 4 drones heterogêneos:** makespan 8,63 min (faixas), 10,63 (k-means),
  8,18 (DARP) e **6,57 min (DARP + refinamento, CV 0,002)**, todos com 100 % de cobertura rasterizada.
- **Benchmark (160 execuções):** DARP + refinamento teve o menor makespan em 31 de 40 casos, a 0,7–0,8 % do
  melhor em média, com CV dos tempos de 0,023–0,028; o k-means não ponderado ficou 50 % acima do melhor com
  frota heterogênea.
- **Colaboração (12 falhas simuladas):** sem realocação perdem-se em média 9,2 % da área (até 26,4 %); a
  repartição recupera a cobertura com +23,1 % de makespan médio (leilão: +31,2 %) e custa 14 quadros LoRa
  (3,8 s no ar em SF9).
- **Detecção de pessoas (VisDrone2019-DET):** o YOLO11s ajustado em 48,7 min no GB10 atingiu mAP50 de
  0,528 no val (COCO: 0,384) e de 0,316 no test-dev (COCO: 0,183). O MAE de contagem foi de 6,91
  pessoas/imagem no val e 8,76 no test-dev, com 25,5 FPS (960 px, FP32, lote 1).

Pesos treinados: `resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt` (19 MB). Não versionados: o
dataset VisDrone (`~/cerise2026/drones/datasets` no Spark, ~4,8 GB extraído) e o `.venv`.

## Validação da contagem em vídeos reais de drone (`validacao_videos/`)

Detector ajustado + ByteTrack/BoT-SORT avaliados no VisDrone2019-MOT (val para escolher parâmetros, test-dev para
reportar): contagem por quadro, pessoas únicas (com a decomposição da inflação por troca de ID), HOTA/MOTA/IDF1 e
FPS no GB10. Resumo do test-dev (ajustado 960 + BoT-SORT): HOTA 0,328, IDF1 0,399, MAE de contagem por quadro 7,46
pessoas; IDs brutos 2.095 para 1.012 pessoas reais. Detalhes, figuras e vídeos em
[`validacao_videos/RELATORIO_SECAO_VIDEOS.md`](validacao_videos/RELATORIO_SECAO_VIDEOS.md) e
[`validacao_videos/README.md`](validacao_videos/README.md).
