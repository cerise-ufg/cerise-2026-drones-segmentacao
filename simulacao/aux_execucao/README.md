# Arquivos auxiliares da execução no DGX Spark

Pasta de trabalho da simulação no Spark (`~/cerise2026/drones/sim/tmp/`), mantida para registro da execução
de 26/10/2026. Não faz parte do pipeline principal: os scripts definitivos estão em `../isaac/`, `../scripts/` e
`../gazebo/`. Os caminhos dentro dos scripts são os do Spark (`~/cerise2026/drones/...`).

| Arquivo(s) | O que é |
|---|---|
| `manifest.sh`, `manifest_arm64.json`, `pull.sh`, `pull_bg.sh`, `pull.log`, `chk_img.sh` | verificação do manifesto arm64 e download da imagem `nvcr.io/nvidia/isaac-sim:5.1.0` (10,1 GB comprimidos) |
| `inspect1.sh`, `inspect2.sh`, `inspect_in.sh`, `ref*.sh`, `ref*_in.sh` | inspeção da imagem do Isaac Sim sem iniciar o app (sem rede, sem GPU); as saídas estão em `../isaac/inspecao/` |
| `instalar.sh`, `instalar2.sh`, `build_contagem.log` | instalação do pacote de simulação no Spark e build da imagem `cerise26_drones_sim` (`../Dockerfile.contagem`) |
| `print_isaac.sh`, `print_isaac.txt`, `print_spark_estado.sh`, `pytest_final.txt` | textos usados nos prints de terminal das evidências |
| `teste_oraculo.*`, `contagem_smoke6.*` | testes da contagem (detector oráculo sem render e smoke test `smoke6`) |
| `fig_teste/` | versões de teste das figuras `sim_09` a `sim_16` |
| `sp/` | scripts de lançamento e acompanhamento das execuções (`lanca_a30.sh`, `lanca_completo.sh`, `cont2.sh`, `cont3.sh`, `varr.sh`, `figs.sh`, `smoke34.sh`, `espera*.sh`, `ver_saida.sh`, `limpa.sh`, `resumo_cont.py`) e os diagnósticos somente leitura do incidente de memória de 14:03 (`ver_kiosk*.sh`) e do dataset do braço (`progresso_braco.sh`) |
| `local/` | quadros de conferência e textos de prints gerados na cópia local das saídas (`debug_mosaico*.jpg`, `video_*.jpg`, `print_*.txt`) |

Os quadros intermediários dos vídeos (`video_run01/`, 866 quadros, e `video_closeup/`, 337 quadros) estão no artefato
`sim_quadros_video.tar` (ver `RELEASES.md` na raiz do repositório); extraído na raiz, ele recria estas pastas.
O pacote `leves.tgz` (cópia dos CSV/JSON de `run01` e `run02a30`) não foi incluído por ser redundante com `../saidas/`.
