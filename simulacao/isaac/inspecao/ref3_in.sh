#!/bin/bash
cd /isaac-sim
echo "=== import NavSchema"; grep -rhn "NavSchema\b" --include=*.py extscache/isaacsim.replicator.agent.core-* extscache/omni.anim.people-* | grep import | sort -u | head
echo "=== get_world_transform / set_world_transform usos"; grep -rn "set_world_transform\|get_world_transform\|anim.graph.core as\|get_character(" --include=*.py extscache/omni.anim.people-* extscache/isaacsim.replicator.agent.core-* extscache/omni.metropolis.utils-* | head -20
echo "=== pyi anim graph"; find extscache/omni.anim.graph.core-* -name "*.pyi" | head; for f in $(find extscache/omni.anim.graph.core-* -name "*.pyi"); do grep -n "def " $f | head -60; done
echo "=== semantics api"; grep -rn "def add_labels\|def add_update_semantics" --include=*.py exts/isaacsim.core.utils extscache/isaacsim.core.utils* 2>/dev/null | head
echo "=== bbox tight fields"; grep -n "bounding_box_2d_tight" -A 3 extscache/omni.replicator.core-*/docs/csv/annotators.csv | head -20
echo "=== ApplyAnimationGraphAPICommand"; grep -rn "class ApplyAnimationGraphAPICommand" -A 12 --include=*.py extscache | head -20
echo "=== get_current_position people"; grep -n "def get_current_position" -A 6 extscache/omni.anim.people-*/omni/anim/people/scripts/character_behavior.py
echo "=== kit python exts para SimulationApp"; grep -n "renderer\|anti_aliasing\|multi_gpu\|\"headless\"" exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py | head -40
