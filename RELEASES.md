# Artefatos grandes (Releases)

Os arquivos abaixo não cabem no git (arquivos acima de 50 MB ou conjuntos grandes de binários) e são publicados
como anexos do Release **`dados-2026-10-26`** deste repositório. Todos são `.tar` sem compressão (o conteúdo é
quase todo JPEG ou NPZ), cada um com menos de 1,9 GB, de modo que não há partes a remontar. Os caminhos dentro de
cada `.tar` já seguem a estrutura do repositório: **extraia na raiz do repositório** e os arquivos caem nas pastas
certas (essas pastas estão no `.gitignore`, então a extração não altera o estado do git).

| Arquivo | Tamanho (bytes) | Arquivos | SHA-256 | Conteúdo |
|---|---|---|---|---|
| `sim_saidas_run01.tar` | 863.815.680 | 4.349 | `2ea6bc0f9a4761fbc917a87bc97a594274abd33830d43481dba7de2f1fbcc8c3` | `simulacao/saidas/run01/` completa: voo a 40 m no Isaac Sim, 866 instantes × 5 câmeras (D1–D4 1280×960/1280×720 e visão geral 1920×1080), `rgb/*.jpg` + poses, caixas de gabarito e parâmetros de câmera; `run01.vigia.log` |
| `sim_saidas_run02a30.tar` | 665.538.560 | 4.021 | `2e7c5bcfeedb4ce43a1eafda9541c2dc8169ed054b3bc0f0997657bef4f9d4aa` | `simulacao/saidas/run02a30/` completa: voo a 30 m, 1.001 instantes × 4 câmeras; `run02a30.vigia.log` |
| `sim_saidas_closeup01.tar` | 531.271.680 | 1.028 | `b6468d6e10a79f6c20597fae8818fd25c9cf8c712663791ccc9710e7fc57adae` | `simulacao/saidas/closeup01/` completa: close-up "cinegrafista", 337 quadros 1920×1080 da câmera do cinegrafista, 337 da câmera do drone (1280×720) e 337 da câmera do drone anotados com gabarito e detecções; quadros de figura (`heroi/`); `closeup01.vigia.log` |
| `sim_saidas_testes.tar` | 115.886.080 | 625 | `08d8242b1363346463bb5bf76d9057037750a99e2f0d616b982d52858130c789` | `simulacao/saidas/` dos testes de fumaça (`smoke`–`smoke6`, `closeup_smoke1`, `closeup_smoke2`), comandos do `run03completo` (interrompido) e os vigias de memória |
| `sim_quadros_video.tar` | 549.754.880 | 1.203 | `467b283a02ab2d755ad641c147733d695e8ff99c569633eb5f4bbe14938cfabc` | `simulacao/aux_execucao/video_run01/` (866 quadros) e `video_closeup/` (337 quadros): quadros compostos que geraram `simulacao/evidencias/sim_video_run01.mp4` e `sim_closeup.mp4` (`scripts/montar_video.sh`) |
| `validacao_videos_cache_det.tar` | 94.064.640 | 96 | `3df22d1ecd68c940248ed62a60386a154771a959b7fc283dcb807ba9c16c54c2` | `validacao_videos/cache_det/`: detecções quadro a quadro (NPZ) no VisDrone2019-MOT val (7 sequências) e test-dev (17) para as 4 variantes (ajustado/COCO × 960/1280 px), entrada do rastreamento (`30_rastrear.py`) |
| `deteccao_runs.tar` | 139.806.720 | 87 | `448a638c165680672640c4f99f19ee646bfe4d5b7154d012f9fb7b7a632eef3b` | `resultados/deteccao/runs/` completa do ultralytics, incluindo os pesos `weights/best.pt` e `last.pt` (19 MB cada) das execuções `ft_pessoa_yolo11s_960` (treino final), `ft_pessoa_yolo11s_960_teste` e `teste960` |

Total: 2.960.138.240 bytes (2,96 GB) em 7 arquivos. As somas também estão em `SHA256SUMS.txt`, anexado ao Release.

O peso do treino final usado em todo o trabalho já está no git:
`resultados/deteccao/best_yolo11s_visdrone_pessoa_960.pt` (19,2 MB, idêntico a
`runs/ft_pessoa_yolo11s_960/weights/best.pt`).

## Como baixar, conferir e extrair

Pela página *Releases* do repositório no GitHub, ou com o GitHub CLI:

```bash
cd cerise-2026-drones-segmentacao             # raiz do repositório clonado
gh release download dados-2026-10-26 -R cerise-ufg/cerise-2026-drones-segmentacao --dir _releases
cd _releases && sha256sum -c SHA256SUMS.txt && cd ..
for f in _releases/*.tar; do tar xf "$f"; done  # extrai na raiz do repositório
```

No Windows (PowerShell), confira cada arquivo com `Get-FileHash -Algorithm SHA256 <arquivo>` e extraia com
`tar -xf <arquivo>` (o `tar` do Windows 10/11) a partir da raiz do repositório.

Os artefatos foram gerados no DGX Spark em 26/10/2026 a partir de `~/cerise2026/drones` com `tar cf` e
`--transform` para mapear `sim/` → `simulacao/`, `sim/tmp/` → `simulacao/aux_execucao/` e `runs/` →
`resultados/deteccao/runs/`.
