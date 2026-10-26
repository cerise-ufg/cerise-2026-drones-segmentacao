# Simulação 3D realista: o que já está pronto (preparo, 26/10/2026)

Objetivo da próxima etapa: rodar no DGX Spark a simulação realista no **Isaac Sim 5.1**, com drones percorrendo
a área aberta do Campus Samambaia/UFG pelas rotas DARP+refinamento e pessoas animadas, e contar as pessoas com o
YOLO11s ajustado. O plano B é **Gazebo Harmonic + PX4 SITL**. A contagem também deve ser validada em vídeos reais de drone.

**Duas condições** precisam estar cumpridas antes de iniciar o Isaac Sim:
- (a) o aceite da licença *NVIDIA Isaac Sim Additional Software and Materials License*
  (<https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/>);
- (b) o término da re-renderização do dataset do braço (contêiner `braco_gen`).

Nesta etapa o Isaac Sim **não foi iniciado** e `ACCEPT_EULA=Y` **não foi passado**. A imagem só foi baixada e
inspecionada com `docker image inspect`/`docker run --entrypoint bash ... ls/cat` (sem rede, sem GPU).

No Spark tudo fica em `~/cerise2026/drones/sim/`. O espelho local está em
`E:\CERISE\FRENTES\NAVEGACAO AUTONOMA\DRONES_SEGMENTACAO\simulacao\` e o código também está no repositório
`cerise-2026-drones-segmentacao`, pasta `simulacao/`.

---

## 1. Isaac Sim 5.1 no DGX Spark: situação medida

| Item | Resultado (medido nesta etapa) |
|---|---|
| Manifesto da `nvcr.io/nvidia/isaac-sim:5.1.0` | tem `linux/arm64` (e `amd64`). **Não exigiu login no NGC.** |
| Download | 17 camadas, 10,1 GB comprimidos, **30 GB em disco**; 22,5 min (10:53→11:15, com a CPU do Spark ocupada) |
| Versão / usuário / entrada | `5.1.0-rc.19+release.26219.9c81211b.gl`; usuário `isaac-sim` (uid **1234**); entrypoint `runheadless.sh` |
| Driver | a imagem exige `MIN_DRIVER_VERSION=570.169`; o Spark tem **580.178.04** (GB10) |
| aarch64 | `runheadless.sh` avisa *"Livestreaming is not supported on aarch64 for 5.1.0"*. Isso não afeta o plano, que usa `python.sh` + `SimulationApp({"headless": True})` sem streaming. |
| Licença / telemetria | `license.sh` bloqueia `runheadless/runapp` sem `ACCEPT_EULA`; `config/privacy.toml` lê `PRIVACY_CONSENT` (com `N`, sem telemetria); `OMNI_ENV_PRIVACY_CONSENT` nunca é definido |
| Disco livre no Spark | 3,2 TB (`/`) |

Inspeção completa: `isaac/inspecao/*.txt`. Print: `evidencias/sim_06_terminal_isaac_imagem.png`.

### Disponível OFFLINE (dentro da imagem)
- Extensões: `omni.anim.people` 0.7.9 (comandos `GoTo/Idle/LookAround`, arquivo de comandos), `isaacsim.replicator.agent.core` 0.7.28 (Actor SDG),
  `omni.anim.navigation.core` 107.3.8 (navmesh: `CreateNavMeshVolumeCommand`, `start_navmesh_baking_and_wait`),
  `omni.anim.graph.core`, `omni.anim.retarget.core`, `omni.replicator.core` 1.12.27 (anotadores `rgb`,
  `bounding_box_2d_tight`, `camera_params`), `isaacsim.core.utils.semantics.add_labels`.
- Materiais MDL locais: `OmniPBR.mdl`, `OmniSurface*.mdl`, `Ground.mdl`. O terreno usa **texturas procedurais próprias**
  (`cerise_sim/texturas.py`): nenhuma imagem de satélite.
- Modelo de quadricóptero local: `extscache/omni.warp.core-*/warp/examples/assets/crazyflie.usd`. O padrão é um quadricóptero procedural.
- Python do kit: numpy, PIL, cv2, scipy, imageio (sem shapely/pyproj; o script do Isaac não precisa deles).

### Exige o servidor de ativos (S3 público configurado em `OMNI_SERVER`; listado só por metadados)
- `Isaac/People/Characters/`: 21 pastas de personagens (25 a 96 MB cada), `Biped_Setup.usd` (grafo de animação) e
  `Isaac/People/Animations/` (`stand_walk_*`, `stand_idle_loop`, `stand_walk_loop_in_place`...). O total de
  `Isaac/People` é 10,0 GB, a maior parte em `DH_Characters`, que não é usado.
- `Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd` e `Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd`.
- Materiais da biblioteca: `NVIDIA/Materials/Base/Natural/{Grass_Countryside,Grass_Cut,Dirt,Asphalt}.mdl`,
  `NVIDIA/Materials/vMaterials_2/{Concrete,Ground}/`, além de `NVIDIA/Assets/{Vegetation,Skies}`.
- Subconjunto necessário: **76 arquivos, 0,45 GB** (0,66 GB com os materiais da biblioteca), calculado com
  `isaac/baixar_ativos.py --listar`. **Não baixado**, porque os ativos estão sob a mesma licença. Depois do aceite há duas opções:
  1. rodar com rede (padrão: rede *bridge* do Docker, sem portas publicadas) e deixar o Isaac baixar os ativos sob demanda (cache em `isaac_cache/`);
  2. fazer o espelho local com `python3 isaac/baixar_ativos.py --destino ~/cerise2026/drones/sim/isaac_ativos` e rodar com
     `ATIVOS_LOCAIS=~/cerise2026/drones/sim/isaac_ativos` (o contêiner sobe com `--network none`).

---

## 2. Cenário (independente do simulador): `cenario/<nome>/`

Gerado por `scripts/gerar_cenario.py` a partir do `cerise_drones`: polígono OSM da área aberta, frota A/B/C/A, DARP com
`refinar=4` e `sementes="centros"`. Tudo em **ENU local** (x leste, y norte, z cima; projeção azimutal equidistante
centrada no centroide), que é o referencial nativo do Isaac Sim (Z para cima, metros) e do Gazebo.

| Cenário | Área | Drones (altitude) | Makespan das trajetórias | Pessoas | Duração | Quadros/câmera a 5 fps |
|---|---|---|---|---|---|---|
| `reduzido` (recorte `-300,-90,110,360`) | 11,27 ha | 4 (40 m) | 170,5 s (D1 168,8 · D2 166,9 · D3 150,6 · D4 170,5) | 80 | 173 s | 866 |
| `reduzido_a30` (mesmo recorte) | 11,27 ha | 4 (**30 m**) | 197,5 s | 80 | 200 s | 1001 |
| `completo` (UFG inteira) | 34,23 ha | 4 (40 m) | 394,1 s (= 6,57 min do plano da frente) | 120 | 397 s | 1986 |

Os tempos das trajetórias temporizadas batem **exatamente** com o modelo `simular_missao` do `cerise_drones` (ver `resumo.json`).

Arquivos de cada cenário:
- `cenario.json`: origem ENU, área e subáreas (ENU), drones (parâmetros, base em ENU e lon/lat, **câmera**, **trajetória
  temporizada** com quadros-chave t/x/y/z/yaw, cobertura/trânsito/curva) e câmera de visão geral.
- `drones/<id>_trajetoria.csv` (10 Hz) e `drones/<id>_camera.csv`: pose da câmera no tempo, com quaternião OpenCV→ENU e
  quaternião USD.
- `pessoas_plano.json`, `pessoas_comandos.txt` (formato do `omni.anim.people`) e `gabarito_pessoas_plano.csv` (verdade-terreno
  t/id/x/y/yaw/andando a 10 Hz). Tipos no `reduzido`: 36 caminhantes (metade pelos caminhos), 14 em grupos andando, 12 em rodas paradas,
  10 parados e 8 que **cruzam a fronteira** entre subáreas (teste de dupla contagem). Velocidades entre 0,8 e 1,7 m/s, com paradas.
- `cena.json`: malhas trianguladas com UV (gramado = polígono OSM; chão externo, rua perimetral de 7 m, calçada e 3 caminhos de
  terra/concreto), 14 edifícios esquemáticos e 78 árvores. O layout é procedural e **não** corresponde aos prédios reais.
- `texturas/*.jpg`: grama, grama seca, terra, concreto, asfalto, telhado e parede (ruído espectral ladrilhável).
- `missoes/*.plan|.kml|.geojson`: missões QGroundControl/PX4 em lon/lat (exportador do `cerise_drones`), usadas no plano B.
- `sem_render/`: execução sintética **sem render** (poses e caixas projetadas de um cilindro de 0,25 m × 1,70 m), no mesmo
  formato da saída dos simuladores.

**Câmeras** (coerentes com o plano da frente de drones): D1, D2 e D4 com HFOV 70° em **1280×960** (4:3, VFOV 55,4° contra 55° do
plano); D3 grande-angular com HFOV 82° em **1280×720** (16:9, VFOV 52,1°). Nadir, com o topo da imagem na direção do voo.
GSD a 40 m: 4,38 cm/px (5,43 no D3). Visão geral: 1920×1080, HFOV 62°, alta a sudoeste da área.
A resolução pode ser aumentada com `--largura` (por exemplo 1920 → 1920×1440), sem mudar o FOV.

Figuras: `evidencias/sim_01_cenario_reduzido.png`, `sim_02_cenario_completo.png` e `sim_03_previa_2d_reduzido.{png,gif}`
(prévia 2D do plano; **não é render**).

---

## 3. Pipeline de contagem (independente do simulador): `cerise_sim/` + `scripts/contar.py`

O fluxo é: quadros + pose → **YOLO11s ajustado em FP32** (`DetectorYOLO`; o FP16 dá NaN no GB10) → projeção de cada caixa no
chão (raio do pixel com os intrínsecos e a pose completa; centro da caixa no plano z = 0,85 m, ou base da caixa em z = 0 para
câmera oblíqua) → **rastreamento no chão** por drone (Húngaro com portão que cresce com o tempo, filtro alfa-beta, faixa
confirmada com ≥ 3 detecções) → **posse da subárea** (a faixa conta para o drone dono, salvo se estiver na faixa de fronteira
de 6 m) → **deduplicação espaço-temporal** (fusão gulosa com restrição "não pode ligar": duas faixas simultâneas do mesmo drone
nunca se juntam). A deduplicação usa extrapolação a velocidade constante, tolerância `raio + v_assoc·Δt` e Δt de até 120 s
para quem está parado e 60 s para quem anda. O resultado é a **contagem única global**.

Também é calculada a agregação estática do `cerise_drones.agregacao.contar` (reaproveitada) sobre as posições das faixas,
para comparação.

A avaliação (`cerise_sim/avaliacao.py`) mede precisão e revocação por quadro contra as caixas de gabarito do Replicator
(IoU 0,5, IoU 0,3 e IoU 0,3 ou centro a até 6 px, para alvos minúsculos), o MAE da contagem por quadro e a contagem única
contra o gabarito, decomposta em clusters corretos, duplicatas, falsos e pessoas perdidas.

**Vídeo real sem pose** (`scripts/contar_video.py`): YOLO → rastreio no plano da imagem → faixas confirmadas. Com anotação no
formato **VisDrone2019-MOT** (pedestrian e people), mede P/R por quadro e o erro da contagem única de IDs.

**Testes: 14 aprovados**, tanto no Windows quanto no Spark, no contêiner `cerise26_drones_sim`
(`evidencias/sim_05_terminal_pytest_spark.png`). Cobrem:
- projeção ida e volta (nadir e oblíqua), convenção de rumo, equivalência com `cerise_drones.agregacao.pixel_para_solo`, USD e quaterniões;
- pegada igual à do `cerise_drones`, e tempo da trajetória igual ao do modelo do `cerise_drones`;
- pessoas dentro da área, rastreadores e casamento P/R;
- ponta a ponta sintético com detector ideal e com detector ruidoso;
- contagem em vídeo no formato VisDrone-MOT.

Resultados medidos **sem render** no `reduzido` (detector oráculo = caixas de gabarito projetadas + ruído; isola o erro da agregação).
Os parâmetros padrão foram escolhidos nesta mesma sensibilidade, portanto o número a reportar é o que sair do render.

| Detector oráculo | Contagem única proposta | Pessoas vistas (cena) | Duplicatas | Falsos | Perdidas | Erro vs. vistas | Soma ingênua de detecções | Faixas sem posse | `cerise_drones` estático |
|---|---|---|---|---|---|---|---|---|---|
| ideal | 82 | 78 (80) | 6 | 0 | 2 | +5,1 % | 1795 | 112 | 78 |
| revocação 0,8, ruído 1 px, 0,3 FP/quadro | 79 | 78 | 4 | 0 | 3 | +1,3 % | 1718 | 109 | 77 |
| revocação 0,6, ruído 2 px, 1 FP/quadro | 86 | 78 | 6 | 6 | 4 | +10,3 % | 2177 | 114 | 81 |

Fontes: `resultados/contagem_oraculo_*.json` e `resultados/sensibilidade_agregacao.json` (42 execuções em 34 s).
Figuras: `evidencias/sim_04_sensibilidade_agregacao.png` e `sim_08_contagem_oraculo_ideal.png`.
Todas as duplicatas com o detector ideal são de pessoas **andando** e revistas depois de 20 a 100 s. Pessoas paradas não geraram
duplicata. Esse é o limite da cobertura assíncrona com alvos móveis.

**Detector no Spark** (`scripts/smoke_yolo.py`, 60 imagens do VisDrone val, imgsz 1280, FP32): **64,7 FPS**, 0 NaN, 30,8
detecções/imagem e pico de 0,17 GB alocado pelo torch, com a GPU compartilhada com a produção
(`resultados/smoke_yolo.json`, `evidencias/sim_07_terminal_smoke_yolo.png`).

---

## 4. Script do Isaac Sim (escrito, NÃO executado): `isaac/sim_ufg_isaac.py`

- Roda em modo standalone com `SimulationApp({"headless": True, "renderer": "RaytracedLighting"})`, sem tela nem streaming.
- Monta o palco a partir de `cena.json`: malhas com UV, OmniPBR com as texturas procedurais (`--materiais biblioteca` usa os
  MDL do servidor), edifícios, árvores, sol (DistantLight, elevação e azimute configuráveis) e céu (DomeLight).
- Pessoas no modo `--modo_pessoas anim_people` (padrão):
  - personagens de `Isaac/People/Characters` em `/World/Characters/Pxxx`, com rótulo semântico `class=person` e NavMeshExcludeAPI;
  - `Biped_Setup` + `populate_anim_graph()`, `ApplyAnimationGraphAPICommand` e o script `character_behavior.py` via ScriptingAPI;
  - arquivo de comandos com as pessoas usadas e navmesh assada num volume que cobre a área;
  - o **gabarito é a posição real** lida do simulador a cada quadro (`omni.anim.graph.core` `get_world_transform`).
- Modo alternativo `cinematico`: posição e rumo do plano via `set_world_transform`, com `Action=Walk` para as pernas. O script cai
  nele sozinho se a navmesh ou o comportamento falharem.
- Drones: câmeras USD com os intrínsecos de `cenario.json` (`focalLength` e aberturas), movidas pela trajetória a cada quadro, e
  modelo visual procedural colorido (padrão escala 5× para aparecer na visão geral; a geometria fica acima da câmera e fora do FOV).
  Opções: `crazyflie_local` e `quadcopter_s3`.
- Gravação: passo de simulação `--dt` 1/30 s e captura a `--fps`, com os *render products* desligados entre capturas.
  `rep.orchestrator.step(rt_subframes=2, delta_time=0, pause_timeline=False)` grava, por drone:
  - `rgb/*.jpg`;
  - `caixas_gt.csv` (`bounding_box_2d_tight`, pessoa identificada pelo `primPath`);
  - `poses.csv`;
  - `camera_params.jsonl`.

  Também grava a visão geral e `gabarito_pessoas.csv`. O `meta.json` registra os segundos por quadro **medidos**.
- Parâmetros: `--duracao --t_inicio --fps --n_pessoas --largura --drones --modo_pessoas --materiais --modelo_drone --renderer
  --rt_subframes --smoke --max_quadros` e iluminação.

`isaac/rodar_isaac.sh`:
- Recusa rodar sem `ACCEPT_EULA=Y` e sem memória *available* ≥ 30 GB, e também com `braco_gen` ativo (a menos de `IGNORAR_BRACO=1`).
- Sobe o contêiner `cerise26_isaac_<nome>` com `--gpus all`, `--cpus 8`, `--cpu-shares 256`, usuário 1234 e rede *bridge* sem portas
  (ou `--network none` com espelho local).
- Monta caches persistentes em `sim/isaac_cache/`, `/sim` somente leitura, `/cenario` somente leitura e `/saida`.
- Liga o **vigia de memória** `isaac/vigia_memoria.sh`, que registra a memória *available* a cada 10 s e **para o contêiner** se ela
  cair abaixo de 10 GB.

### Comandos exatos (próxima etapa, no Spark)
```bash
# pré-condições
free -g                                     # "available" >= 30 GB
sudo docker ps --format '{{.Names}}' | grep -x braco_gen || echo "braco terminou"
sudo docker ps --format '{{.Names}} {{.Status}}' | grep vllm   # produção no ar (não tocar)

# 1) smoke test: 1 drone, 6 pessoas, 4 s. A 1ª vez compila shaders e baixa ~0,45 GB de ativos.
ACCEPT_EULA=Y PRIVACY_CONSENT=N bash ~/cerise2026/drones/sim/isaac/rodar_isaac.sh smoke --smoke
sudo docker logs -f cerise26_isaac_smoke
#    conferir em ~/cerise2026/drones/sim/saidas/smoke/:
#      meta.json (s_por_quadro, modo_pessoas, avisos), D1/rgb/*.jpg, D1/caixas_gt.csv não vazio,
#      gabarito_pessoas.csv, log_sim.txt (navmesh assada: True)
#    e a memória máxima em ~/cerise2026/drones/sim/saidas/smoke.vigia.log

# 2) render do cenário reduzido (4 drones, 80 pessoas, 173 s, 5 fps, 866 quadros por câmera)
ACCEPT_EULA=Y PRIVACY_CONSENT=N bash ~/cerise2026/drones/sim/isaac/rodar_isaac.sh run01 --cenario /cenario/reduzido --fps 5
#    variantes: --cenario /cenario/reduzido_a30 (pessoas maiores na imagem) | --largura 1920 | --modo_pessoas cinematico

# 3) contagem com o YOLO11s ajustado (FP32) e avaliação contra o gabarito
cd ~/cerise2026/drones/sim
bash scripts/rodar_contagem_spark.sh cont_run01 scripts/contar.py --execucao saidas/run01 --cenario cenario/reduzido \
     --detector yolo --pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --imgsz 1280 \
     --saida resultados/contagem_run01.json --figura

# 4) vídeo do relatório (mosaico visão geral + 4 drones, gabarito em verde, detecções em vermelho)
bash scripts/rodar_contagem_spark.sh video_run01 scripts/video_relatorio.py --execucao saidas/run01 \
     --deteccoes resultados/contagem_run01_deteccoes.csv --saida evidencias/video_run01.mp4
```

### Memória e tempo esperados (ESTIMATIVAS, não medidas; o smoke test mede `s_por_quadro`)
- **Memória:** cerca de 15 a 20 GB de memória unificada para o reduzido (Kit + RTX + 5 *render products* de até 1920×1080 +
  80 personagens com ~35 MB de ativos cada). Por isso o script exige 30 GB *available* para iniciar, e o vigia para tudo abaixo de
  10 GB. Se faltar, reduza `--n_pessoas`, use `--sem_visao_geral`, rode com `--drones D1,D2` e depois `D3,D4`, ou diminua `--largura`.
- **Tempo:** 0,5 a 1,5 s por quadro capturado (5 câmeras, `rt_subframes=2`, ~6 passos de animação por captura):
  - reduzido (866 capturas): **~7 a 22 min**, mais 5 a 15 min na 1ª vez (shaders, ativos, navmesh);
  - completo (1986 capturas): ~17 a 50 min.

  Para calibrar: extrapole a partir de `s_por_quadro` do smoke e dos logs `quadro N ... estimativa total`.
- **Disco:** cerca de 2 GB por execução do reduzido (JPEG 1280×960 ~0,3 MB × 4 × 866 + visão geral).

---

## 5. Plano B: Gazebo Harmonic + PX4 SITL (`gazebo/`, preparado, nada compilado)

- `Dockerfile`: `ubuntu:22.04` arm64, Gazebo Harmonic (repositório oficial OSRF) + `python3-gz-transport13/gz-msgs10`, PX4
  `v1.15.4` (GitHub oficial) compilado com `cmake -DCONFIG=px4_sitl_default` + `nice -n 19 ninja -j $JOBS`, e MAVSDK-Python.
- `gerar_mundo.py` (só numpy, já testado: gera `mundo_ufg.sdf` válido para o `reduzido`, com 80 atores, 4 drones, 14 edifícios e
  78 árvores):
  - coordenadas esféricas na origem ENU; terreno em OBJ com as mesmas texturas;
  - **atores** com `walk.dae`/`stand.dae` (Gazebo Fuel) seguindo **as mesmas trajetórias temporizadas** (gabarito exato) e
    `Label` único por pessoa;
  - `x500_cerise_<id>`: o x500 do PX4 com **câmera em nádir** e `boundingbox_camera` 2D com os intrínsecos do drone;
  - câmera estática de visão geral.
- `rodar_sim_gz.sh` (dentro do contêiner):
  - `gz sim -s -r --headless-rendering` (EGL, **sem DISPLAY**);
  - N instâncias `px4 -i k` com `PX4_GZ_MODEL_NAME`;
  - `capturar.py` (gz-transport: imagem + pose do x500 interpolada, **atitude completa**, + caixas por rótulo, no mesmo formato
    de saída do Isaac);
  - `rodar_missoes.py` (MAVSDK: importa os `.plan` do `cerise_drones`, envia, arma e inicia as missões juntas).
- `rodar_gazebo.sh` (host): checa memória e `braco_gen`, gera o mundo e sobe o contêiner `cerise26_gzpx4_<run>` com o vigia de memória.
  Há também um `docker-compose.yml` equivalente.

```bash
cd ~/cerise2026/drones/sim/gazebo
sudo nice -n 19 docker build --build-arg JOBS=4 -t cerise26_gzpx4:latest . > ../logs/build_gzpx4.log 2>&1
bash ~/cerise2026/drones/sim/gazebo/rodar_gazebo.sh gz_run01 reduzido 5
# contagem: igual ao passo 3 com --execucao saidas/gz_run01
```

**Estimativa de build** (não medida):
- downloads: ~1,5 a 2 GB de apt + ~1 GB do clone do PX4 com submódulos;
- compilação do SITL: ~8 a 12 min com `JOBS=16` e CPU livre, ou ~30 a 45 min com `JOBS=4` e o Spark carregado;
- imagem final: ~6 a 8 GB.

Execução: provavelmente mais lenta que o tempo real com 80 atores e 5 câmeras (PX4 em *lockstep*), algo como ~6 a 12 min para o reduzido.

---

## 6. Vídeos reais de drone

A **validação completa em vídeo real** foi feita em `../validacao_videos/`:
- dados: VisDrone2019-MOT val (7 sequências) e test-dev (17), que ficam no Spark em `~/cerise2026/drones/mot`;
- rastreamento: ByteTrack e BoT-SORT;
- métricas: HOTA, MOTA e IDF1, e erro da contagem única;
- relatório: `RELATORIO_SECAO_VIDEOS.md`.

Os números de referência estão naquele relatório. O `scripts/contar_video.py` daqui é a versão leve e testada do **mesmo pipeline** da simulação
(rastreador `RastreadorImagem`, formato VisDrone-MOT). Ele serve para conferir um vídeo novo, por exemplo um voo próprio no
Campus Samambaia (`--video voo.mp4`), ou uma sequência do MOT:
`scripts/rodar_contagem_spark.sh mot_<seq> scripts/contar_video.py --quadros /w/mot/.../sequences/<seq> --gt /w/mot/.../annotations/<seq>.txt
--pesos /w/resultados/best_yolo11s_visdrone_pessoa_960.pt --saida resultados/mot_<seq>.json`.

---

## 7. Riscos e pontos de atenção

1. **APIs não exercitadas.** O script do Isaac foi escrito a partir das extensões e exemplos **da própria imagem 5.1**, mas não foi
   executado. Pontos a verificar no smoke test:
   - ordem de atualização entre o grafo de animação e o `set_world_transform` (modo cinemático);
   - `NavMeshExcludeAPI` / volume da navmesh numa área de ~450 × 420 m;
   - `primPaths` no `bounding_box_2d_tight`;
   - exposição da luz (`--sol_intensidade`, `--ceu_intensidade`).
2. **RTX/driver no GB10.** Há relatos de falha no RTX init da 6.0.x com drivers novos; a 5.1 é a mais documentada para o DGX Spark.
   Se o smoke falhar no RTX init, registrar o log (`sim/isaac_cache/logs`) e partir para o plano B.
3. **Pessoas minúsculas em nádir a 40 m.** No centro da imagem a largura de uma pessoa é ~11 px. Mediana do maior lado da caixa
   projetada: 26,7 px (p10 16,9). O detector foi ajustado no VisDrone, que tem vistas majoritariamente oblíquas (diferença de domínio).
   Alternativas: `reduzido_a30` (mediana 36,1 px), `--largura 1920`, ou câmera oblíqua (`gerar_cenario.py --inclinacao 20` + contagem com `--ponto_ref base`).
4. **Pessoas andando + cobertura assíncrona** geram duplicatas e fusões (medido acima). O número a reportar é o do render, com os
   parâmetros fixados antes (sensibilidade sem render).
5. **Memória unificada** compartilhada com a produção (vLLM ~55 %): manter ≥ 10 GB *available* (o vigia para o contêiner sozinho).
6. **Realismo fotométrico** das texturas procedurais: alternativa com `--materiais biblioteca`.
7. **Plano B:** atores do Gazebo com a mesma malha (`walk.dae`) para todos; `boundingbox_camera` com atores precisa ser conferido.
   Se não rotular os atores, usar as caixas projetadas pelo `cerise_sim.sintetico.caixa_pessoa` (sem oclusão).

## 8. Onde está cada coisa (Spark: `~/cerise2026/drones/`)
- `sim/cerise_sim/`: pacote (enu, camera, trajetoria, pessoas, cena, texturas, dados, rastreador, contagem, avaliacao, sintetico)
- `sim/scripts/`: gerar_cenario, figura_cenario, previa_2d, contar, contar_video, sensibilidade_agregacao, figura_sensibilidade,
  video_relatorio, smoke_yolo, rodar_contagem_spark.sh, rodar_testes_spark.sh
- `sim/isaac/`: sim_ufg_isaac.py, rodar_isaac.sh, vigia_memoria.sh, baixar_ativos.py, inspecao/; `sim/isaac_inspecao/` tem as
  saídas brutas e `ref/` (cópias de referência da API, fora do git)
- `sim/gazebo/`: Dockerfile, docker-compose.yml, gerar_mundo.py, rodar_sim_gz.sh, capturar.py, rodar_missoes.py, rodar_gazebo.sh
- `sim/cenario/{reduzido,reduzido_a30,completo}/`, `sim/resultados/`, `sim/evidencias/`, `sim/tests/`
- `cerise_drones/` e `dados/`: cópia do pacote da frente (usada pelo gerador e pela agregação)
- Imagens Docker: `nvcr.io/nvidia/isaac-sim:5.1.0` (30 GB) e `cerise26_drones_sim:latest` (= `cerise26_drones` + shapely, pyproj, lap, pytest).
  A `cerise26_gzpx4` ainda não foi construída.

---

## 9. Execução (26/10/2026, tarde): feita

A simulação foi executada depois do aceite da licença (`ACCEPT_EULA=Y`, `PRIVACY_CONSENT=N`) e do fim do `braco_gen`.
Resultados, correções de API e limitações estão em [`RELATORIO_SECAO_SIMULACAO.md`](RELATORIO_SECAO_SIMULACAO.md). Resumo das mudanças em relação a este preparo:
- espelho local de ativos (83 arquivos, 0,55 GB), incluindo `Isaac/People/Characters/biped_demo/` (sem ele o grafo de animação não compila);
- volume da navmesh definido direto (extensão + translação): o `TransformPrim` não tinha efeito e a navmesh ficava com ±4,5 m;
- modo `anim_people` validado (pessoas seguem o plano); `--camera_debug`, `--depurar` e `--pessoas` para testes;
- contagem com BoT-SORT (`cerise_sim/botsort_chao.py`) + varredura (`scripts/varredura_botsort.py`); 17 testes;
- `rodar_isaac.sh` com `--memory 24g` e vigia a cada 2 s com `docker kill`: o cenário `completo` provocou OOM global na assadura da navmesh (1.330 × 998 m) e **não deve ser rodado assim de novo**.
