#!/bin/bash
# copia arquivos de referencia (API) da imagem para /out/ref (sem iniciar o app)
O=/out/ref; mkdir -p $O; cd /isaac-sim
IRA=$(ls -d extscache/isaacsim.replicator.agent.core-*); PPL=$(ls -d extscache/omni.anim.people-*); NAV=$(ls -d extscache/omni.anim.navigation.core-*)
for f in $IRA/isaacsim/replicator/agent/core/stage_util.py $IRA/isaacsim/replicator/agent/core/simulation.py $IRA/isaacsim/replicator/agent/core/agent_manager.py \
         $IRA/isaacsim/replicator/agent/core/settings.py $IRA/isaacsim/replicator/agent/core/config_file/default.py $IRA/isaacsim/replicator/agent/core/config_file/defines.py \
         $IRA/isaacsim/replicator/agent/core/data_generation/data_generation.py $IRA/isaacsim/replicator/agent/core/randomization/character_randomizer.py \
         $IRA/data/config_file/config_file_default_values.yaml $IRA/config/default_config.yaml $IRA/config/default_command.txt $IRA/docs/README.md $IRA/test_data/test_config_file.yaml $IRA/test_data/test_command.txt \
         $PPL/docs/README.md $PPL/omni/anim/people/settings.py $PPL/omni/anim/people/scripts/character_behavior.py $PPL/omni/anim/people/scripts/commands/goto.py \
         $PPL/omni/anim/people/scripts/commands/idle.py $PPL/omni/anim/people/scripts/commands/look_around.py $PPL/omni/anim/people/scripts/commands/base_command.py \
         $PPL/omni/anim/people/scripts/utils.py $PPL/omni/anim/people/scripts/navigation_manager.py $PPL/omni/anim/people/python_ext.py \
         standalone_examples/testing/omni.replicator.agent/test_scripting.py standalone_examples/testing/isaacsim.ros2.bridge/test_people_sim.py \
         standalone_examples/api/isaacsim.replicator.examples/multi_camera.py standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py \
         standalone_examples/api/isaacsim.replicator.examples/custom_fps_writer_annotator.py standalone_examples/benchmarks/benchmark_sdg.py; do
  [ -f "$f" ] && cp "$f" "$O/$(echo $f | tr '/' '__' | sed 's/extscache__//')" ; done
find $NAV -name "*.py" | head -40 > $O/nav_py_list.txt
grep -rn "def \|bake\|start_navmesh\|NavMeshVolume\|query_navmesh" --include=*.py $NAV | grep -v tests | head -120 > $O/nav_api.txt
grep -rn "def " $NAV/omni/anim/navigation/core/*.py 2>/dev/null | head -60 >> $O/nav_api.txt
ls $NAV/omni/anim/navigation/core/ >> $O/nav_api.txt
# kit apps usados pelo SimulationApp
grep -n "omni.anim\|replicator\|people" apps/isaacsim.exp.base.python.kit apps/isaacsim.exp.full.kit apps/isaacsim.exp.action_and_event_data_generation.base.kit | head -60 > $O/kit_apps.txt
cat isaac-sim.action_and_event_data_generation.sh >> $O/kit_apps.txt
ls -la $O | wc -l
