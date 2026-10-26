# Saídas das execuções do Isaac Sim (metadados)

Cópia de `~/cerise2026/drones/sim/saidas/` no DGX Spark **sem os quadros JPEG**. Para cada execução:

| Arquivo | Conteúdo |
|---|---|
| `meta.json` | parâmetros da execução e tempos medidos (`s_por_quadro`, `s_render_por_quadro`, `tempo_laco_s`, avisos, fonte do gabarito) |
| `gabarito_pessoas.csv` | posição real de cada pessoa a cada quadro, lida do grafo de animação |
| `<drone>/poses.csv`, `<drone>/camera_params.jsonl` | pose e parâmetros da câmera por quadro (D1–D4, `visao_geral`, `debug`) |
| `<drone>/caixas_gt.csv` | caixas 2D de gabarito do Replicator (`bounding_box_2d_tight`) |
| `pessoas_comandos_usados.txt`, `log_sim.txt` | comandos do `omni.anim.people` usados e log do script |
| `../<execução>.vigia.log` | memória *available* do Spark e do contêiner registrada pelo vigia |
| `closeup01/quadros.csv`, `closeup01/heroi/*.png` | plano quadro a quadro do close-up e os quadros de figura em alta qualidade |

Execuções: `smoke`–`smoke6` e `closeup_smoke1`/`closeup_smoke2` (testes de fumaça), `run01` (voo a 40 m),
`run02a30` (voo a 30 m), `closeup01` (close-up) e `run03completo` (cenário completo, interrompido pelo incidente de
memória; só os comandos e o vigia).

Os quadros brutos (`*/rgb/*.jpg` e `closeup01/gimbal/anotado/*.jpg`) estão nos artefatos `sim_saidas_run01.tar`,
`sim_saidas_run02a30.tar`, `sim_saidas_closeup01.tar` e `sim_saidas_testes.tar` (ver `RELEASES.md` na raiz).
Extraídos na raiz do repositório, eles recriam esta pasta completa.
