# Validação da contagem de pessoas em vídeos reais de drone

Mede quão bem o detector YOLO11s ajustado no VisDrone-DET (`resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt`),
combinado com rastreamento multiobjeto (ByteTrack e BoT-SORT do ultralytics), **conta pessoas em vídeo real de
drone com gabarito** — VisDrone2019-MOT, partições val (7 sequências, 2.846 quadros) e test-dev (17 sequências,
6.635 quadros). A seção de relatório está em [`RELATORIO_SECAO_VIDEOS.md`](RELATORIO_SECAO_VIDEOS.md).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `scripts/00_baixar_visdrone_mot.py` | download resumível dos zips oficiais (links do Google Drive do README do VisDrone) |
| `scripts/01_preparar_libs.sh` | `lap`, `motmetrics` (PyPI) num diretório montado, sem alterar a imagem `cerise26_drones` |
| `scripts/run_vv.sh` | executa um script no contêiner `cerise26_drones` (destacado, `nice 19`, `--cpus`, GPU opcional) |
| `scripts/10_detectar.py` | detecção quadro a quadro em cache (4 variantes: ajustado/COCO × 960/1280 px, FP32) |
| `scripts/20_gmc_condicoes.py` | GMC (sparseOptFlow) pré-calculada por sequência + atributos de condição |
| `scripts/30_rastrear.py` | grade no val, escolha de parâmetros, avaliação no test-dev, trilhas em formato MOTChallenge |
| `scripts/metricas_mot.py` | HOTA, CLEAR (MOTA) e Identity (IDF1) — reimplementação do TrackEval — e erros de contagem |
| `scripts/31_conferir_metricas.py` | testes sintéticos + conferência de MOTA/IDF1 contra o py-motmetrics |
| `scripts/32_ablacao_gmc.py` | BoT-SORT com e sem compensação de movimento de câmera |
| `scripts/40_fps.py` | vazão do pipeline em fluxo no GB10 (leitura + detecção + rastreador) |
| `scripts/50_evidencias.py`, `51_mosaico.py`, `52_erro_unico_vs_L.py`, `53_analise_erros.py` | figuras, tabela por sequência, análises |
| `scripts/60_montar_videos.sh` | MP4 H.264 (ffmpeg/libx264 do host) a partir dos quadros renderizados |
| `resultados/` | todas as métricas medidas (`.json`/`.csv`) |
| `evidencias/` | figuras `v00`–`v14` e os vídeos `v_video_*.mp4` |

## Como reproduzir (DGX Spark)

```bash
cd ~/cerise2026/drones/validacao_videos/scripts
nice -n 19 python3 -I 00_baixar_visdrone_mot.py            # -> ~/cerise2026/drones/mot (7,4 GB extraído; fora do git)
bash 01_preparar_libs.sh
CPUS=6 bash run_vv.sh det 10_detectar.py                    # GPU, ~8 min, pico 1,0 GB alocado
GPU=0 CPUS=4 bash run_vv.sh gmc 20_gmc_condicoes.py         # CPU
GPU=0 CPUS=4 bash run_vv.sh rastreio 30_rastrear.py         # CPU, ~3 min (34 configs x 8 combinações no val)
GPU=0 CPUS=2 bash run_vv.sh conferencia 31_conferir_metricas.py
GPU=0 CPUS=4 bash run_vv.sh gmcabl 32_ablacao_gmc.py
CPUS=4 bash run_vv.sh fps 40_fps.py                         # GPU
GPU=0 CPUS=4 bash run_vv.sh evid 50_evidencias.py && bash 60_montar_videos.sh
GPU=0 CPUS=2 bash run_vv.sh mosaico 51_mosaico.py; GPU=0 CPUS=2 bash run_vv.sh erros 53_analise_erros.py
```

Também versionados: a GMC pré-calculada (`cache_gmc/`, 0,5 MB), as trilhas de todas as combinações
(`rastreios/<detector>_<rastreador>/<split>/<seq>.txt`, formato MOTChallenge, ~38 MB) e os logs das execuções
no Spark (`logs/`). O cache de detecções (`cache_det/`, 90 MB) está no artefato `validacao_videos_cache_det.tar`
(ver `RELEASES.md` na raiz). O VisDrone2019-MOT (dados de terceiros, 7,4 GB extraídos) não é redistribuído:
`scripts/00_baixar_visdrone_mot.py` o baixa dos links oficiais.
