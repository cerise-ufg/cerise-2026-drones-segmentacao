#!/bin/bash
cd /isaac-sim
NAV=$(ls -d extscache/omni.anim.navigation.core-*); PPL=$(ls -d extscache/omni.anim.people-*)
echo "=== command.py"; sed -n 40,110p $NAV/omni/anim/navigation/core/scripts/command.py
echo "=== bindings"; ls $NAV/omni/anim/navigation/core/bindings/ 2>/dev/null; find $NAV -name "*.pyi" | head; for f in $(find $NAV -name "*.pyi"); do grep -n "def " $f | head -60; done
echo "=== usos de baking"; grep -rn "start_navmesh_baking\|CreateNavMeshVolumeCommand\|NavMeshVolume.Define\|navmesh_volume\|setNavMeshVolumeType\|NavMeshVolumeType\|GetNavMeshVolumeTypeAttr" --include=*.py extscache exts 2>/dev/null | grep -v "/tests/" | head -30
echo "=== populate_anim_graph"; sed -n 1,60p $PPL/omni/anim/people/scripts/custom_command/populate_anim_graph.py | grep -n "def \|import\|Biped\|anim_graph" | head -30
echo "=== semantics util IRA"; grep -rn "def add_update_prim_metrosim_semantics" -A 25 extscache/omni.metropolis.utils-*/ | head -40
