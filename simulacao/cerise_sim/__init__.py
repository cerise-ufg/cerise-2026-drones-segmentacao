"""cerise_sim: preparo da simulacao 3D (Isaac Sim / Gazebo+PX4) e pipeline de contagem de pessoas.

Modulos independentes do simulador:
- enu        : lon/lat e UTM <-> coordenadas locais metricas ENU (x=leste, y=norte, z=cima)
- camera     : intrinsecos (pinhole), pose de camera em nadir/obliqua, projecao mundo<->pixel
- trajetoria : trajetorias temporizadas dos drones a partir das rotas do cerise_drones
- pessoas    : plano de pessoas (caminhantes, grupos, parados) e verdade-terreno no tempo
- cena       : layout procedural do terreno (caminhos, edificios esquematicos)
- dados      : formato comum do dataset gravado (Isaac ou Gazebo) e leitores
- rastreador : rastreamento multi-alvo no plano do chao (Hungaro) e no plano da imagem
- contagem   : pipeline deteccao -> rastreio -> projecao -> posse/deduplicacao -> contagem global
- avaliacao  : precisao/revocacao por quadro e erro da contagem unica global
"""
__version__ = "0.1.0"
