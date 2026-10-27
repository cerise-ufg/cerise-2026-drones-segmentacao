# Drones: segmentação de área, detecção e contagem de pessoas e simulação

Roteiro de implementação da frente de drones do CERISE/EMC/UFG (Plano de Trabalho 2026). Os caminhos citados são relativos à raiz deste repositório.

## O que será reproduzido

Este roteiro reproduz a frente de drones do Plano de Trabalho 2026: a divisão de uma área georreferenciada entre drones heterogêneos, com as missões de cobertura (Objetivo 5, Atividade 13); a realocação da área quando um drone abandona a missão (algoritmos de colaboração); e a detecção e contagem de pessoas em imagens aéreas (Objetivo 5, item 1), validada em vídeos reais e numa simulação 3D no NVIDIA Isaac Sim 5.1. O código está no repositório `cerise-2026-drones-segmentacao` da organização `cerise-ufg`.

Ao final você terá:

- o pacote `cerise_drones` com as missões de 4 drones em `.plan` (QGroundControl), KML e GeoJSON e o mapa em HTML;
- o benchmark dos métodos de partição e a simulação de falhas com realocação e custo LoRa;
- um YOLO11s ajustado para "pessoa" no VisDrone2019-DET, com mAP, erro de contagem e FPS;
- as métricas de rastreamento e de contagem única no VisDrone2019-MOT;
- o voo simulado de 4 drones sobre 11,3 ha do Campus Samambaia com 80 pessoas animadas, a contagem única global e o vídeo;
- o plano B Gazebo + PX4 pronto para construir (não executado no projeto).

## Requisitos

**Tabela: Máquinas usadas no projeto e alternativas**

| Etapa | Configuração usada | Alternativa |
|---|---|---|
| Segmentação, benchmark e colaboração | Windows 11 ou Linux, só CPU | qualquer computador com Python 3.11 |
| Detecção, vídeos reais e contagem | NVIDIA DGX Spark (GPU GB10 Blackwell sm_121, CPU aarch64, 128 GB de memória unificada), Ubuntu 24.04, Docker + NVIDIA Container Toolkit | x86_64 com GPU NVIDIA e o mesmo Docker (não testado) |
| Simulação 3D | a mesma estação, `nvcr.io/nvidia/isaac-sim:5.1.0` (arm64) | x86_64 com GPU NVIDIA RTX; a imagem também é publicada para amd64 (não testado) |

Software:

- Python 3.11 (testado com 3.11.9), com as bibliotecas nas versões do ambiente usado no projeto, fixadas no comando de instalação do passo 1 da segmentação (numpy 2.4.6, shapely 2.2.0, pyproj 3.7.2, scikit-learn 1.9.1, scipy 1.17.1, matplotlib 3.11.2, folium 0.20.0, imageio 2.38.1, pillow 13.0.0, playwright 1.64.1, pandas 3.1.0 e pytest 9.1.1);
- Docker com `--gpus all`, `sudo`, `git`, GitHub CLI (`gh`), `curl` e `ffmpeg` com libx264 no host;
- `nvcr.io/nvidia/pytorch:26.05-py3` (PyTorch 2.12) com ultralytics 8.4.189 ([`deteccao/logs/build.log`](deteccao/logs/build.log)); como o [`deteccao/Dockerfile`](deteccao/Dockerfile) instala a versão mais recente, troque `ultralytics` por `ultralytics==8.4.189` para fixá-la;
- Isaac Sim 5.1.0, que exige driver NVIDIA 570.169 ou mais novo (o projeto usou o 580.178.04).

Acessos: leitura na organização `cerise-ufg` do GitHub (repositórios privados; solicite ao bolsista); internet para o VisDrone, os pesos COCO e os ativos do Isaac; e o aceite, por você mesmo, da *NVIDIA Isaac Sim Additional Software and Materials License* (https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/).

Disco: cerca de 100 GB (imagem do Isaac 30 GB, imagens próprias dezenas de GB, VisDrone-DET 4,8 GB, VisDrone-MOT 7,4 GB, Releases 2,96 GB, 0,8 GB por voo simulado). Tempos medidos: benchmark 584 s, download do VisDrone-DET 10 min, treino 48,7 min, detecção e rastreamento nos vídeos 11 min, download da imagem do Isaac 22,5 min, render de cada voo 21 min. Conte de um a dois dias.

## Obtenção do código e dos dados

1. Clone o repositório. Os scripts de Docker usam a pasta fixa `~/cerise2026/drones` (montada como `/w`) e chamam a pasta da simulação de `sim`; clone nesse caminho e crie o link relativo, que vale dentro e fora dos contêineres. O link e o que os scripts criam (como `logs/`, `home/` e `pylib_mot/`) aparecem como não rastreados no `git status`; não os versione.

```bash
gh auth login
mkdir -p ~/cerise2026/hf_cache
gh repo clone cerise-ufg/cerise-2026-drones-segmentacao ~/cerise2026/drones
cd ~/cerise2026/drones && ln -s simulacao sim
```

2. Baixe os Releases que quiser (nenhum é obrigatório). O tag publicado é `dados-2026-10-26`. `sim_saidas_run01.tar` e `sim_saidas_run02a30.tar` trazem os quadros dos voos renderizados, `validacao_videos_cache_det.tar` as detecções dos vídeos e `deteccao_runs.tar` as `runs/` do treino. Use `-p <arquivo>` para baixar só alguns; extraia na raiz do repositório:

```bash
gh release download dados-2026-10-26 -R cerise-ufg/cerise-2026-drones-segmentacao \
  -D ~/cerise2026/_releases
cd ~/cerise2026/_releases && sha256sum -c --ignore-missing SHA256SUMS.txt
cd ~/cerise2026/drones && for f in ~/cerise2026/_releases/*.tar; do tar xf "$f"; done
```

3. O peso final do detector já está no git ([`resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt`](resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt)). O VisDrone (dados de terceiros, uso acadêmico) e os ativos do Isaac Sim (licença da NVIDIA) não são redistribuídos: você os baixa nas etapas a seguir.

## Segmentação da área e geração das missões

1. Crie o ambiente e rode os testes (no Windows, use `.venv\Scripts\python`). O pacote não tem instalador; os scripts põem a raiz do repositório no `sys.path`.

```bash
cd ~/cerise2026/drones
python3.11 -m venv .venv
.venv/bin/python -m pip install numpy==2.4.6 shapely==2.2.0 pyproj==3.7.2 \
  scikit-learn==1.9.1 scipy==1.17.1 matplotlib==3.11.2 folium==0.20.0 \
  imageio==2.38.1 pillow==13.0.0 playwright==1.64.1 pandas==3.1.0 pytest==9.1.1
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -q tests
```

2. Rode a demonstração. Ela divide o campo aberto do Campus Samambaia ([`dados/ufg_samambaia_campo_aberto.geojson`](dados/ufg_samambaia_campo_aberto.geojson), 34,23 ha, OpenStreetMap) entre 4 drones dos tipos A, B, C e A ([`cerise_drones/frota.py`](cerise_drones/frota.py)) por faixas, k-means, DARP e DARP + refinamento, e exporta o plano final em [`resultados/missao_ufg/`](resultados/missao_ufg) (`.plan`, `.kml` e `.geojson` por drone e `mapa_ufg_darp_ref.html`), além das figuras `evidencias/01` a `03`:

```bash
.venv/bin/python scripts/demo_ufg.py
```

3. Para uma área sua, crie `minha_missao.py` na raiz. Zonas proibidas entram em `AreaMissao.de_arquivos(nome, area, [zonas], margem_zona_m=10)`; bases em lon/lat são convertidas com `area.proj.ponto_xy(lon, lat)`. Com o polígono da UFG, o exemplo imprime `makespan 6.57 min  CV 0.002  cobertura 100.00 %` e grava 13 arquivos e o mapa em `saida_missao/`:

```python
from cerise_drones import AreaMissao, criar_frota, planejar_missao
from cerise_drones.cenarios import bases_no_contorno
from cerise_drones.visualizar import mapa_folium

area = AreaMissao.de_arquivos("minha area",
                              "dados/ufg_samambaia_campo_aberto.geojson")
frota = criar_frota(["A", "B", "C", "A"], bases_no_contorno(area, 4), altitude_m=40)
plano = planejar_missao(area, frota, "darp", refinar=4, sementes="centros")
m = plano.metricas
print(f"makespan {m['makespan_s'] / 60:.2f} min  CV {m['cv_tempo']:.3f}  "
      f"cobertura {m['cobertura_pct']:.2f} %")
plano.exportar("saida_missao", "missao")
mapa_folium(plano, "saida_missao/mapa.html", "Minha missao")
```

4. Rode o benchmark (4 cenários, 2 a 6 drones, frotas homogênea e heterogênea), a ablação, os gráficos, as falhas com realocação (repartição e leilão) e a análise LoRa. Eles regravam [`resultados/`](resultados) e [`evidencias/`](evidencias); compare com `git diff --stat`.

```bash
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark.py --ablacao
.venv/bin/python scripts/graficos_benchmark.py
.venv/bin/python scripts/colaboracao_sim.py
.venv/bin/python scripts/protocolo_analise.py
```

![Plano esperado: DARP + refinamento no Campus Samambaia com 4 drones heterogêneos a 40 m, com subáreas, rotas e bases](evidencias/02_ufg_plano_darp_ref.png)

*Plano esperado: DARP + refinamento no Campus Samambaia com 4 drones heterogêneos a 40 m, com subáreas, rotas e bases*

## Detector de pessoas YOLO11s no VisDrone2019-DET

Rode na estação com GPU, em `~/cerise2026/drones`. O [`deteccao/run.sh`](deteccao/run.sh) sobe o contêiner `cerise26_drones_<nome>` em segundo plano, com log em `logs/<nome>.log`; espere cada passo com `sudo docker wait`.

1. Construa a imagem `cerise26_drones:latest`. Em x86_64 o Dockerfile deve servir, pois a imagem NGC também existe para essa arquitetura, mas isso não foi testado.

```bash
bash deteccao/build.sh
```

2. Baixe os zips oficiais (URLs do `VisDrone.yaml` do ultralytics) e prepare a classe única "pessoa" (pedestrian + people), com o treino reduzido a 960 px:

```bash
NP=12 bash deteccao/00_fetch_visdrone_parallel.sh
bash deteccao/run.sh prep 02_prep_pessoa.py
sudo docker wait cerise26_drones_prep && tail -3 logs/prep.log
```

3. Teste a precisão numérica. No GB10 o FP16 gerou NaN; por isso os scripts usam FP32 (`amp=False`, `half=False`). Em outra GPU, rode este teste antes de ativar FP16:

```bash
bash deteccao/run.sh amp teste_amp.py
```

4. Treine. O treino é limitado pelo relógio (`--horas`) e por um teto de memória de GPU (`--teto_gb`, padrão 7,5 GB). O projeto fez 5 épocas em 48,7 min, com pico de 6,5 GB; numa GPU mais lenta, aumente `--horas`. A opção `--teste` faz 1 época em 5 % do treino; `--workers 8` repete a execução registrada em `deteccao/logs/treino.log` (o padrão do script é 6).

```bash
bash deteccao/run.sh treino 04_treinar.py --modelo yolo11s.pt --imgsz 960 \
  --batch 4 --horas 0.8 --workers 8
sudo docker wait cerise26_drones_treino
```

5. Se pulou o treino, copie o peso versionado para o caminho que a avaliação e as etapas seguintes esperam:

```bash
mkdir -p runs/ft_pessoa_yolo11s_960/weights
cp resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt \
  runs/ft_pessoa_yolo11s_960/weights/best.pt
```

6. Avalie o YOLO11s COCO e o ajustado (mAP no val e no test-dev, erro de contagem e FPS). A avaliação exige o peso do passo 4 ou do passo 5: sem ele, o `06_avaliar_tudo.py` avalia só o modelo COCO, sem aviso. A saída vai para `resultados/avaliacao_deteccao.json`; a de referência está em [`resultados/deteccao/`](resultados/deteccao).

```bash
bash deteccao/run.sh avaliacao 06_avaliar_tudo.py
```

![Exemplos esperados do detector ajustado no VisDrone2019-DET, com a contagem real e a detectada no limiar 0,175](evidencias/15_deteccao_exemplos.png)

*Exemplos esperados do detector ajustado no VisDrone2019-DET, com a contagem real e a detectada no limiar 0,175*

## Contagem em vídeos reais (VisDrone2019-MOT)

1. Baixe o val e o test-dev pelos links oficiais do Google Drive (o script retoma o download e espera quando o Drive responde *Quota exceeded*; os SHA-256 estão na seção 2 de [`validacao_videos/RELATORIO_SECAO_VIDEOS.md`](validacao_videos/RELATORIO_SECAO_VIDEOS.md)) e instale `lap`, `motmetrics` e `xmltodict` em `pylib_mot/`:

```bash
cd ~/cerise2026/drones/validacao_videos/scripts
nice -n 19 python3 -I 00_baixar_visdrone_mot.py
bash 01_preparar_libs.sh
```

2. Rode o pipeline: [`10_detectar.py`](validacao_videos/scripts/10_detectar.py) guarda em cache as detecções de 4 variantes (ajustado e COCO, 960 e 1280 px, FP32); [`20_gmc_condicoes.py`](validacao_videos/scripts/20_gmc_condicoes.py) calcula a compensação de movimento de câmera (GMC); [`30_rastrear.py`](validacao_videos/scripts/30_rastrear.py) escolhe os parâmetros do ByteTrack e do BoT-SORT numa grade no val e os aplica sem alteração ao test-dev; [`32_ablacao_gmc.py`](validacao_videos/scripts/32_ablacao_gmc.py) tira a GMC; [`40_fps.py`](validacao_videos/scripts/40_fps.py) mede a vazão. O peso ajustado deve estar em `runs/ft_pessoa_yolo11s_960/weights/best.pt`; o `weights/yolo11s.pt` é baixado pelo ultralytics.

```bash
CPUS=6 bash run_vv.sh det 10_detectar.py
sudo docker wait cerise26_drones_vv_det
GPU=0 CPUS=4 bash run_vv.sh gmc 20_gmc_condicoes.py
sudo docker wait cerise26_drones_vv_gmc
GPU=0 CPUS=4 bash run_vv.sh rastreio 30_rastrear.py
sudo docker wait cerise26_drones_vv_rastreio
GPU=0 CPUS=4 bash run_vv.sh gmcabl 32_ablacao_gmc.py
sudo docker wait cerise26_drones_vv_gmcabl
CPUS=4 bash run_vv.sh fps 40_fps.py
```

Os logs ficam em `~/cerise2026/drones/logs/vv_<nome>.log` e os resultados regravam [`validacao_videos/resultados/`](validacao_videos/resultados). Com o Release `validacao_videos_cache_det.tar`, pule o [`10_detectar.py`](validacao_videos/scripts/10_detectar.py); a GMC já está versionada em [`validacao_videos/cache_gmc/`](validacao_videos/cache_gmc).

## Simulação 3D no Isaac Sim 5.1

1. Leia e aceite a licença do Isaac Sim (ver Requisitos); só então passe `ACCEPT_EULA=Y`. `PRIVACY_CONSENT=N` desliga a telemetria.

2. Baixe a imagem oficial (não exigiu login no NGC), construa a imagem da contagem e rode os testes do pacote `cerise_sim`:

```bash
sudo docker pull nvcr.io/nvidia/isaac-sim:5.1.0
cd ~/cerise2026/drones/sim
sudo nice -n 19 docker build -f Dockerfile.contagem -t cerise26_drones_sim:latest .
bash scripts/rodar_testes_spark.sh
```

3. Faça o espelho local dos ativos oficiais (personagens, animações, `Biped_Setup.usd`, esqueleto `biped_demo` e quadricóptero); `--listar` só mostra os tamanhos:

```bash
python3 isaac/baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos --listar
python3 isaac/baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos
```

4. Use os cenários versionados [`cenario/reduzido`](simulacao/cenario/reduzido) (recorte de 11,27 ha, 4 drones a 40 m nas rotas DARP + refinamento, 80 pessoas, 173 s) e [`cenario/reduzido_a30`](simulacao/cenario/reduzido_a30) (o mesmo a 30 m). Para criar outro, use [`scripts/gerar_cenario.py`](simulacao/scripts/gerar_cenario.py) com o ambiente da segmentação; este comando refaz o `reduzido_a30` com outro nome:

```bash
../.venv/bin/python scripts/gerar_cenario.py --nome meu_a30 \
  --recorte=-300,-90,110,360 --altitude 30
```

5. Rode o teste de fumaça (1 drone, 6 pessoas, 4 s; a primeira execução, com a compilação de shaders, levou cerca de 2 min). O [`isaac/rodar_isaac.sh`](simulacao/isaac/rodar_isaac.sh) exige `ACCEPT_EULA=Y` e 30 GB de memória *available* (`MIN_LIVRE_INICIO_GB`), limita o contêiner a 24 GB (`MEM_LIMITE`) e 8 CPUs (`CPUS`) e liga o vigia [`isaac/vigia_memoria.sh`](simulacao/isaac/vigia_memoria.sh), que mata o contêiner abaixo de 12 GB *available* (`MIN_LIVRE_GB`). Ele também recusa iniciar se houver um contêiner `braco_gen` (outro trabalho do projeto), o que não deve ocorrer na sua máquina.

```bash
export ACCEPT_EULA=Y PRIVACY_CONSENT=N
export ATIVOS_LOCAIS=$HOME/cerise2026/drones/sim/isaac_ativos
bash isaac/rodar_isaac.sh meu_smoke --smoke
sudo docker logs -f cerise26_isaac_meu_smoke
```

6. Renderize os dois voos com nomes novos: [`saidas/run01`](simulacao/saidas/run01) e [`saidas/run02a30`](simulacao/saidas/run02a30) guardam os metadados do projeto, e o script passa a pasta de saída para o usuário do contêiner (uid 1234).

```bash
bash isaac/rodar_isaac.sh meu_run01 --cenario /cenario/reduzido --fps 5
sudo docker wait cerise26_isaac_meu_run01
bash isaac/rodar_isaac.sh meu_run02a30 --cenario /cenario/reduzido_a30 --fps 5 \
  --sem_visao_geral
```

7. Conte as pessoas. O [`scripts/contar.py`](simulacao/scripts/contar.py) detecta com o YOLO11s ajustado (FP32, 1280 px), rastreia com cinco variantes (BoT-SORT recomendado pelos vídeos reais, BoT-SORT calibrado `botsort_cal`, rastreador próprio no solo, BoT-SORT com GMC pela pose e sem GMC), projeta as trilhas no solo e deduplica pela posição, não pelos IDs. Para 30 m, troque `meu_run01` por `meu_run02a30` e o cenário por [`cenario/reduzido_a30`](simulacao/cenario/reduzido_a30):

```bash
bash scripts/rodar_contagem_spark.sh cont_meu_run01 scripts/contar.py \
  --execucao saidas/meu_run01 --cenario cenario/reduzido --detector yolo \
  --pesos /w/resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
  --rastreadores botsort,botsort_cal,chao,botsort_pose,botsort_semgmc \
  --saida resultados/contagem_meu_run01.json --figura
sudo docker wait cerise26_drones_cont_meu_run01 && grep '^\[' logs/cont_meu_run01.log
```

Sem renderizar, reproduza a contagem do projeto com o cache versionado: repita o comando com `--execucao saidas/run01`, `--cache_det resultados/contagem_run01_cache_det.npz` e `--saida resultados/repro_contagem_run01.json`.

8. Gere o vídeo (visão geral, quatro câmeras com gabarito em verde e trilhas em vermelho, contagem acumulada) e monte o MP4 no host:

```bash
bash scripts/rodar_contagem_spark.sh video_meu_run01 scripts/video_relatorio.py \
  --execucao saidas/meu_run01 --contagem resultados/contagem_meu_run01.json \
  --deteccoes resultados/contagem_meu_run01_deteccoes_botsort_cal.csv \
  --quadros_dir tmp/video_meu_run01
sudo docker wait cerise26_drones_video_meu_run01
bash scripts/montar_video.sh tmp/video_meu_run01 evidencias/meu_video_run01.mp4 15
```

Em x86_64 com GPU dedicada (não testado), as verificações do script medem só a RAM do host; o uso de memória de vídeo não foi medido.

![Cena esperada no Isaac Sim (t = 103,4 s): visão geral com os quatro drones e câmera do D4 em nádir a 40 m com as caixas de gabarito do Replicator](simulacao/evidencias/sim_09_cena_render.png)

*Cena esperada no Isaac Sim (t = 103,4 s): visão geral com os quatro drones e câmera do D4 em nádir a 40 m com as caixas de gabarito do Replicator*

## Plano B: Gazebo Harmonic + PX4 SITL

O plano B foi preparado, mas não foi construído nem executado; trate-o como não testado. O [`simulacao/gazebo/Dockerfile`](simulacao/gazebo/Dockerfile) parte do `ubuntu:22.04`, instala o Gazebo Harmonic e compila o PX4 v1.15.4 (`px4_sitl_default`); como usa `dpkg --print-architecture`, deve servir também em x86_64. O `rodar_gazebo.sh` regrava o mundo versionado [`gazebo/mundo_reduzido/`](simulacao/gazebo/mundo_reduzido) com o `gerar_mundo.py` (imagem `cerise26_drones`) e sobe o `gz sim` sem tela, uma instância PX4 por drone, a captura das câmeras (`capturar.py`) e as missões `.plan` via MAVSDK (`rodar_missoes.py`):

```bash
cd ~/cerise2026/drones/sim/gazebo && mkdir -p ../logs
sudo nice -n 19 docker build --build-arg JOBS=4 -t cerise26_gzpx4:latest . \
  > ../logs/build_gzpx4.log 2>&1
bash rodar_gazebo.sh gz_run01 reduzido 5
```

A saída tem o formato da do Isaac; conte com o passo 7 acima e `--execucao saidas/gz_run01`. As estimativas da seção 5 de [`simulacao/PRONTO.md`](simulacao/PRONTO.md) não foram medidas.

## Verificação do resultado

**Tabela: Resultados de referência medidos no projeto**

| Etapa | O que deve aparecer | Arquivo de referência |
|---|---|---|
| Testes | `16 passed` (pacote) e `17 passed` (simulação) | [`resultados/pytest_log.txt`](resultados/pytest_log.txt), [`simulacao/aux_execucao/pytest_final.txt`](simulacao/aux_execucao/pytest_final.txt) |
| Demonstração UFG | DARP + refinamento: makespan 6,57 min, CV 0,002, 11,37 km, cobertura 100 %; 13 arquivos exportados | [`resultados/demo_ufg_log.txt`](resultados/demo_ufg_log.txt), [`resultados/demo_ufg_metricas.json`](resultados/demo_ufg_metricas.json) |
| Benchmark | DARP + refinamento a 0,7-0,8 % do melhor makespan em média, 31 vitórias em 40 casos | [`resultados/benchmark_resumo.md`](resultados/benchmark_resumo.md) |
| Colaboração | sem realocação perdem-se 9,2 % da área em média; a repartição recupera a cobertura com +23,1 % de makespan | [`resultados/colaboracao.csv`](resultados/colaboracao.csv) |
| Detector | mAP50 0,528 no val (COCO 0,384) e 0,316 no test-dev; MAE de contagem 6,91 (val) e 8,76 (test-dev); 25,5 FPS a 960 px | [`resultados/deteccao/avaliacao_deteccao.json`](resultados/deteccao/avaliacao_deteccao.json) |
| Vídeos reais | ajustado 960 + BoT-SORT no test-dev: HOTA 0,328, IDF1 0,399, MAE 7,46; 2.095 IDs para 1.012 pessoas | [`validacao_videos/resultados/metricas_agregadas.json`](validacao_videos/resultados/metricas_agregadas.json) |
| Isaac Sim | `navmesh assada: True`; voo a 40 m com 866 quadros a 1,46 s/quadro e 11,5 GB no contêiner | [`simulacao/resultados/logs/isaac_smoke6.log`](simulacao/resultados/logs/isaac_smoke6.log), [`simulacao/resultados/logs/isaac_run01.log`](simulacao/resultados/logs/isaac_run01.log), [`simulacao/saidas/run01.vigia.log`](simulacao/saidas/run01.vigia.log) |
| Contagem simulada | AP50 0,702 (40 m) e 0,652 (30 m); `botsort_cal`: contagem única de 78 para 79 pessoas vistas (40 m; 13 duplicatas e 14 perdidas) e de 70 para 73 (30 m; 16 duplicatas e 19 perdidas) | [`simulacao/resultados/logs/cont_run01.log`](simulacao/resultados/logs/cont_run01.log), [`simulacao/resultados/logs/cont_run02a30.log`](simulacao/resultados/logs/cont_run02a30.log), [`simulacao/resultados/contagem_run01.json`](simulacao/resultados/contagem_run01.json), [`simulacao/resultados/contagem_run02a30.json`](simulacao/resultados/contagem_run02a30.json) |

Com o cache versionado, a contagem deve repetir os números do log. Num render novo, compare a ordem de grandeza: a reprodutibilidade bit a bit do render não foi verificada.

## Problemas comuns e soluções

- **NaN em FP16 no GB10.** Com o PyTorch 2.12 do NGC 26.05, o FP16 gerou NaN e 263 caixas onde o FP32 achava 14 ([`deteccao/logs/amp.log`](deteccao/logs/amp.log)). Mantenha `amp=False` e `half=False`.
- **Contagem inflada.** Não conte pelos IDs do rastreador: nos vídeos foram 2.095 IDs para 1.012 pessoas e, na simulação, a soma de detecções deu 1.306 para 79 pessoas. Use a deduplicação no solo do `contar.py`.
- **Subcontagem no simulado.** O BoT-SORT recomendado pelos vídeos contou 34 de 79, porque o `fuse_score` multiplica a IoU por pontuações baixas; use o `botsort_cal`. Sem GMC (`botsort_semgmc`), nada é rastreado a 5 fps.
- **Falta de memória.** A navmesh do cenário `completo` (1.330 × 998 m) consumiu mais de 34 GB em cerca de 22 s e causou OOM global ([`simulacao/resultados/logs/oom_run03completo_dmesg.txt`](simulacao/resultados/logs/oom_run03completo_dmesg.txt)). Use o cenário reduzido e mantenha `MEM_LIMITE` e o vigia.
- **Personagens em pose fixa e gabarito em (0, 0).** Com o aviso `Type mismatch for variable Action/Walk`, falta o esqueleto `biped_demo`; refaça o espelho com o `baixar_ativos.py` atual, que já o inclui.
- **Pessoas andando em linha reta em direções arbitrárias.** A navmesh ficou pequena; confira no `log_sim.txt` a linha `navmesh assada: True` com volume de cerca de 452 × 467 m.
- **Peso não encontrado.** A avaliação e os vídeos esperam `runs/ft_pessoa_yolo11s_960/weights/best.pt`; copie o peso versionado (passo 5 da detecção).

## Referências do repositório

- [`README.md`](README.md), [`RELEASES.md`](RELEASES.md) e [`RELATORIO_SECAO.md`](RELATORIO_SECAO.md): visão geral, artefatos grandes e métodos;
- [`cerise_drones/`](cerise_drones) (`geo.py`, `frota.py`, `particao.py`, `cobertura.py`, `planejador.py`, `exportar.py`, `colaboracao.py`, `protocolo.py`) e [`scripts/`](scripts);
- [`deteccao/Dockerfile`](deteccao/Dockerfile), [`deteccao/run.sh`](deteccao/run.sh), [`deteccao/04_treinar.py`](deteccao/04_treinar.py), [`deteccao/05_avaliar_ft.py`](deteccao/05_avaliar_ft.py);
- [`validacao_videos/README.md`](validacao_videos/README.md) e [`validacao_videos/RELATORIO_SECAO_VIDEOS.md`](validacao_videos/RELATORIO_SECAO_VIDEOS.md);
- [`simulacao/PRONTO.md`](simulacao/PRONTO.md) e [`simulacao/RELATORIO_SECAO_SIMULACAO.md`](simulacao/RELATORIO_SECAO_SIMULACAO.md): preparo, correções de API e resultados;
- [`simulacao/isaac/sim_ufg_isaac.py`](simulacao/isaac/sim_ufg_isaac.py), [`simulacao/isaac/rodar_isaac.sh`](simulacao/isaac/rodar_isaac.sh), [`simulacao/isaac/vigia_memoria.sh`](simulacao/isaac/vigia_memoria.sh), [`simulacao/scripts/contar.py`](simulacao/scripts/contar.py), [`simulacao/cerise_sim/botsort_chao.py`](simulacao/cerise_sim/botsort_chao.py) e [`simulacao/gazebo/`](simulacao/gazebo).
