# cerise-2026-drones-segmentacao

Segmentação da área de atuação entre drones, colaboração na missão, mensagens LoRa e detecção e contagem
de pessoas em imagens aéreas, com validação em vídeos reais de drone e simulação 3D no NVIDIA Isaac Sim.

> Autor: **Eduardo Martins de Carvalho Caixeta** (bolsista), CERISE/EMC/UFG, 2026.
> Coordenação da área de Navegação Autônoma: Prof. Dr. Alisson Assis Cardoso.

## Atividades do Plano de Trabalho CERISE 2026 cobertas

| Plano | O que foi feito | Onde |
|---|---|---|
| Objetivo 5, Atividade 13: software funcional de segmentação da área de atuação entre os drones disponíveis | pacote `cerise_drones` (faixas, k-means/Voronoi, DARP e DARP + refinamento por tempo), cobertura boustrophedon por subárea com bateria e zonas proibidas, exportação QGroundControl `.plan`/KML/GeoJSON, benchmark e ablação | `cerise_drones/`, `scripts/`, `resultados/`, `RELATORIO_SECAO.md` |
| Objetivo 5, item 1: planejamento e adequação de um projeto com drones para detecção e contagem de pessoas | detector YOLO11s ajustado no VisDrone2019-DET (classe "pessoa") no DGX Spark, agregação e deduplicação das contagens entre drones, requisitos de hardware | `deteccao/`, `resultados/deteccao/`, `RELATORIO_SECAO.md` (seções 4.5, 4.6 e 5) |
| Objetivo 2: algoritmos de colaboração entre drones | falha ou bateria baixa no meio da missão: repartição da área órfã e leilão, com custo de comunicação LoRa | `cerise_drones/colaboracao.py`, `cerise_drones/protocolo.py`, `scripts/colaboracao_sim.py` |
| Validação da contagem em vídeos reais de drone | YOLO11s + ByteTrack/BoT-SORT no VisDrone2019-MOT (HOTA, MOTA, IDF1, contagem única, FPS no GB10) | `validacao_videos/` |
| Simulação 3D realista | Isaac Sim 5.1 headless no DGX Spark: 4 drones nas rotas DARP + refinamento sobre 11,3 ha do Campus Samambaia, 80 pessoas animadas, gabarito do Replicator, contagem única global; close-up "cinegrafista"; plano B Gazebo + PX4 preparado | `simulacao/` |

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `cerise_drones/` | pacote Python: `geo`, `frota`, `particao`, `cobertura`, `planejador`, `metricas`, `exportar`, `visualizar`, `colaboracao`, `protocolo` (quadros `cerise_proto` v1 + extensão), `agregacao`, `cenarios` |
| `dados/` | polígonos do Campus Samambaia/UFG extraídos do OpenStreetMap (ODbL) |
| `scripts/` | demonstração na UFG, benchmark, ablação, gráficos, colaboração, protocolo e agregação |
| `tests/` | 16 testes do pacote (`python -m pytest -q tests`) |
| `deteccao/` | `Dockerfile` da imagem `cerise26_drones` (`FROM nvcr.io/nvidia/pytorch:26.05-py3` + ultralytics), `build.sh`, `run.sh`, download e preparação do VisDrone2019-DET, avaliação COCO, fine-tuning, avaliação final; `config/visdrone_pessoa.yaml`; `logs/` (logs brutos do build, download, treino e avaliação no Spark) |
| `resultados/` | métricas medidas (`.csv`/`.json`/`.md`), missão exportada da UFG (`missao_ufg/`), detecção (`deteccao/`: pesos ajustados `best_yolo11s_visdrone_pessoa_960.pt`, métricas, curvas e as `runs/` do ultralytics sem os arquivos de pesos) |
| `evidencias/` | figuras da seção de relatório |
| `validacao_videos/` | scripts (`00`–`60`), resultados, evidências (figuras e 2 vídeos MP4), GMC pré-calculada (`cache_gmc/`), trilhas de todas as combinações (`rastreios/`), logs; ver `validacao_videos/README.md` |
| `simulacao/` | pacote `cerise_sim`, cenários (`cenario/`), scripts do Isaac Sim (`isaac/`) e do plano B (`gazebo/`, com o mundo gerado `mundo_reduzido/`), close-up (`closeup/`), resultados e logs (`resultados/`), metadados das execuções (`saidas/`), evidências (figuras e vídeos), 17 testes (`tests/`), arquivos auxiliares da execução (`aux_execucao/`), `PRONTO.md` (preparo) e `RELATORIO_SECAO_SIMULACAO.md` |
| `ferramentas/` | `term2png.py`, que gera os prints de terminal das evidências a partir dos logs |
| `RELATORIO_SECAO.md` | seção do relatório técnico (segmentação, colaboração, LoRa, detecção, agregação) |
| `RELEASES.md` | artefatos grandes publicados como Releases (tamanho, SHA-256, conteúdo e como extrair) |

## Como reproduzir

### Planejamento (Windows ou Linux, Python 3.11)

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy shapely pyproj scikit-learn scipy matplotlib folium imageio pillow playwright pandas pytest
.venv/bin/python -m playwright install chromium     # só para capturar o mapa HTML
.venv/bin/python scripts/demo_ufg.py                # UFG: 4 drones, exporta .plan/.kml/.geojson + mapa
.venv/bin/python scripts/benchmark.py               # 4 cenários x N=2..6 x 2 frotas x 4 métodos
.venv/bin/python scripts/benchmark.py --ablacao
.venv/bin/python scripts/graficos_benchmark.py
.venv/bin/python scripts/colaboracao_sim.py         # falha/bateria baixa + realocação (GIF)
.venv/bin/python -m pytest -q tests
```

No Windows, use `.venv\Scripts\python`. Uso como biblioteca:

```python
from cerise_drones import AreaMissao, Drone, planejar_missao
area = AreaMissao.de_arquivos("minha area", "area.geojson", ["zona_proibida.geojson"], margem_zona_m=10)
frota = [Drone("D1", velocidade_ms=10, autonomia_min=20, hfov_graus=70, vfov_graus=55, altitude_m=40), ...]
for d, (lon, lat) in zip(frota, bases): d.partida_xy = area.proj.ponto_xy(lon, lat)
plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
plano.exportar("saida", "missao")      # missao_D1.plan (QGroundControl), .kml, .geojson
```

### Detecção (DGX Spark, GB10 aarch64, Docker + nvidia-container-toolkit)

O código é copiado para `~/cerise2026/drones` (mesma estrutura deste repositório).

```bash
cd ~/cerise2026/drones && bash deteccao/build.sh               # imagem cerise26_drones:latest
NP=12 bash deteccao/00_fetch_visdrone_parallel.sh              # zips oficiais (github.com/ultralytics/assets)
bash deteccao/run.sh prep 02_prep_pessoa.py                     # pedestrian+people -> "pessoa"
bash deteccao/run.sh treino 04_treinar.py --modelo yolo11s.pt --imgsz 960 --batch 4 --horas 0.8
bash deteccao/run.sh avaliacao 06_avaliar_tudo.py               # mAP COCO x ajustado, contagem, FPS
```

No GB10 (PyTorch 2.12, NGC 26.05) o autocast FP16 produziu NaN (`deteccao/teste_amp.py`); treino e inferência
foram feitos em FP32 (`amp=False`, `half=False`).

### Vídeos reais e simulação

- Vídeos reais (VisDrone2019-MOT): passo a passo em `validacao_videos/README.md`.
- Simulação: imagem `cerise26_drones_sim` (`simulacao/Dockerfile.contagem`, derivada de `cerise26_drones`), Isaac Sim
  `nvcr.io/nvidia/isaac-sim:5.1.0` (exige aceitar a licença da NVIDIA, aceita pelo Eduardo em 26/10/2026; usado com `ACCEPT_EULA=Y` e `PRIVACY_CONSENT=N`)
  e os comandos de `simulacao/PRONTO.md` (seção "Comandos exatos") e de `simulacao/RELATORIO_SECAO_SIMULACAO.md`:

```bash
# no Spark (~/cerise2026/drones/sim)
python3 isaac/baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos            # ativos oficiais (0,55 GB)
ACCEPT_EULA=Y PRIVACY_CONSENT=N ATIVOS_LOCAIS=~/cerise2026/drones/sim/isaac_ativos \
  bash isaac/rodar_isaac.sh run01 --cenario /cenario/reduzido --fps 5
bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido \
  --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,botsort_cal,chao,botsort_pose,botsort_semgmc --saida resultados/contagem_run01.json --figura
```

As saídas brutas das execuções (quadros JPEG) e os caches grandes estão nos Releases (ver `RELEASES.md`). Para
recompor a árvore completa, extraia cada `.tar` na raiz do repositório.

## Resultados principais

Todos os números saem dos arquivos em `resultados/`, `validacao_videos/resultados/` e `simulacao/resultados/`.

**Segmentação, Campus Samambaia/UFG (34,23 ha, 4 drones heterogêneos;** `resultados/demo_ufg_metricas.json`**):**

| Método | Makespan (min) | CV do tempo | Distância total (km) | Cobertura (%) |
|---|---|---|---|---|
| Faixas ponderadas | 8,63 | 0,120 | 13,36 | 100,00 |
| k-means + Voronoi | 10,63 | 0,385 | 10,65 | 100,00 |
| DARP (sementes nas bases) | 8,18 | 0,152 | 11,66 | 100,00 |
| **DARP + refinamento** | **6,57** | **0,002** | 11,37 | 100,00 |

- **Benchmark (160 execuções;** `resultados/benchmark_resumo.csv`**):** o DARP + refinamento teve o menor makespan em 31
  de 40 casos, a 0,7–0,8 % do melhor em média, com CV dos tempos de 0,023–0,028; o k-means não ponderado ficou 50 % acima
  do melhor com frota heterogênea.
- **Colaboração (12 falhas simuladas;** `resultados/colaboracao.csv`**):** sem realocação perdem-se em média 9,2 % da área
  (até 26,4 %); a repartição recupera a cobertura com +23,1 % de makespan médio (leilão: +31,2 %) e custa 14 quadros
  LoRa (3,8 s no ar em SF9).
- **Detecção de pessoas, VisDrone2019-DET (**`resultados/deteccao/avaliacao_deteccao.json`**):** o YOLO11s ajustado em
  48,7 min no GB10 atingiu mAP50 de 0,528 no val (COCO: 0,384) e de 0,316 no test-dev (COCO: 0,183); MAE de contagem de
  6,91 pessoas/imagem no val e 8,76 no test-dev; 25,5 FPS (960 px, FP32, lote 1).
- **Vídeos reais, VisDrone2019-MOT test-dev (ajustado 960 + BoT-SORT;** `validacao_videos/resultados/metricas_agregadas.json`**):**
  HOTA 0,328, IDF1 0,399, MAE de contagem por quadro de 7,46 pessoas; 2.095 IDs brutos para 1.012 pessoas reais.
- **Simulação no Isaac Sim (**`simulacao/resultados/contagem_run01.json`, `contagem_run02a30.json`**):** AP50 por quadro
  de 0,702 (40 m) e 0,652 (30 m). Contagem única global contra as pessoas distintas vistas (79 a 40 m, 73 a 30 m):

| Rastreador + deduplicação no solo | 40 m | 30 m |
|---|---|---|
| BoT-SORT recomendado pelos vídeos reais (0,40, `fuse_score`) | 34 (−57,0 %) | 29 (−60,3 %) |
| rastreador próprio no solo (conf ≥ 0,175) | 88 (+11,4 %) | 94 (+28,8 %) |
| BoT-SORT sem `fuse_score`, limiar 0,15 (escolhido a 40 m) | 78 (−1,3 %) | 70 (−4,1 %; 54 corretos, 16 duplicatas, 19 perdidas) |

- **Close-up (**`simulacao/resultados/closeup_yolo_gimbal.json`**):** 337 quadros 1920×1080 em 9,5 min no Spark, pico de
  12,35 GiB no contêiner; YOLO11s na câmera do drone com AP50 de 0,568.

Detalhes, figuras e limitações: `RELATORIO_SECAO.md`, `validacao_videos/RELATORIO_SECAO_VIDEOS.md` e
`simulacao/RELATORIO_SECAO_SIMULACAO.md`.

## Artefatos em Releases

Arquivos grandes ou conjuntos grandes de binários ficam fora do git e são publicados como Releases do repositório,
com SHA-256: saídas brutas das execuções do Isaac Sim (`run01`, `run02a30`, `closeup01` e testes), quadros
intermediários dos vídeos, cache de detecções dos vídeos reais e as `runs/` completas do treino (com os pesos
`best.pt`/`last.pt` de cada execução). Tabela completa e instruções em [`RELEASES.md`](RELEASES.md).

## O que não está aqui e por quê

| Item | Motivo | Como obter |
|---|---|---|
| VisDrone2019-DET (`datasets/`, 4,8 GB) e VisDrone2019-MOT (`mot/`, 7,4 GB) | dados de terceiros (equipe AISKYEYE, Universidade de Tianjin), distribuídos para uso acadêmico, sem licença de redistribuição | DET: `deteccao/00_fetch_visdrone_parallel.sh` (zips de `github.com/ultralytics/assets`, *release* `v0.0.0`) + `deteccao/02_prep_pessoa.py`; MOT: `validacao_videos/scripts/00_baixar_visdrone_mot.py` (links oficiais de `github.com/VisDrone/VisDrone-Dataset`; SHA-256 dos zips em `validacao_videos/RELATORIO_SECAO_VIDEOS.md`) |
| Anotações convertidas do VisDrone (`datasets/visdrone_pessoa/gt_*.json`, `labels/`) | derivadas dos dados de terceiros | regeneradas por `deteccao/02_prep_pessoa.py` |
| Pesos pré-treinados `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo26n.pt` e a fonte `Arial.ttf` do ultralytics | arquivos de terceiros (Ultralytics, AGPL-3.0) | baixados automaticamente pelo ultralytics 8.4.189 dos *releases* de `github.com/ultralytics/assets` |
| Ativos do Isaac Sim (`simulacao/isaac_ativos/`, 0,66 GB: personagens, animações, materiais, céus HDRI) e os HDRIs processados `simulacao/closeup/ceu/*.hdr` | ativos distribuídos pela NVIDIA (licença do Isaac Sim) | `simulacao/isaac/baixar_ativos.py` (e `--closeup`); `simulacao/closeup/ceu.sh` regenera os HDRIs sem o sol (parâmetros medidos em `closeup/ceu/*.json`) |
| Trechos de código e configuração das extensões da NVIDIA copiados na inspeção (`isaac_inspecao/ref/`, `isaac/inspecao/06`, `08`, `12`, `13`) | código de terceiros | `simulacao/aux_execucao/ref*_in.sh` e `simulacao/aux_execucao/inspect_in.sh` refazem a inspeção dentro da imagem |
| Imagens `nvcr.io/nvidia/isaac-sim:5.1.0` (30 GB) e `nvcr.io/nvidia/pytorch:26.05-py3` | imagens Docker de terceiros | `docker pull` do NGC |
| Imagens próprias `cerise26_drones`, `cerise26_drones_sim` e `cerise26_gzpx4` | dezenas de GB, regeneráveis | `deteccao/Dockerfile` (`deteccao/build.sh`), `simulacao/Dockerfile.contagem`, `simulacao/gazebo/Dockerfile` |
| Caches (`simulacao/isaac_cache/` 1,3 GB, `home/.nv/ComputeCache` 0,35 GB), `pylib_mot/`, `.venv/`, `__pycache__/` | regeneráveis | criados na primeira execução; `pylib_mot/` por `validacao_videos/scripts/01_preparar_libs.sh` (lap 0.5.13, motmetrics 1.4.0, xmltodict 1.0.4, do PyPI) |
| `simulacao/aux_execucao/leves.tgz` | cópia redundante dos metadados de `run01`/`run02a30` | já incluídos em `simulacao/saidas/` |

Nenhum segredo (chave, token ou senha) faz parte deste trabalho.

## Pendências

- Hardware: construção dos módulos LoRa definitivos, adequação dos drones e instalação dos módulos (requisitos na
  seção 5 de `RELATORIO_SECAO.md`).
- Abrir os `.plan` de `resultados/missao_ufg/` no QGroundControl e executar em SITL PX4 + Gazebo (só a validação
  estrutural foi feita; o plano B Gazebo + PX4 está preparado e não foi compilado).
- Incorporar ao `cerise_proto` (projeto LoRa) a extensão AREA_POLY, TASK_BID e TASK_AWARD definida em `cerise_drones/protocolo.py`.
- Detecção: treinar mais épocas, testar 1280 px e TensorRT, ajustar com imagens próprias do Campus Samambaia em nádir e
  calibrar as pontuações no domínio simulado.
- Renderizar o cenário `completo` (34,23 ha) com a navmesh por subárea; validar em voo real no Campus Samambaia.

## Licença

Este repositório não inclui arquivo de licença, por decisão do autor: todos os direitos são reservados ao autor e ao
CERISE/EMC/UFG. Os dados e componentes de terceiros citados acima seguem as licenças dos respectivos detentores.
