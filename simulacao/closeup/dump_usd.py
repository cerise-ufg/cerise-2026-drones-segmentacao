import sys
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(sys.argv[1])
print("upAxis", UsdGeom.GetStageUpAxis(st), "mpu", UsdGeom.GetStageMetersPerUnit(st), "default", st.GetDefaultPrim())
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
for p in st.Traverse():
    extra = ""
    if p.IsA(UsdGeom.Gprim):
        bb = cache.ComputeWorldBound(p).ComputeAlignedRange()
        extra = f" bb={tuple(round(v,3) for v in bb.GetMin())}..{tuple(round(v,3) for v in bb.GetMax())}"
    refs = p.GetMetadata("references")
    print(str(p.GetPath()), p.GetTypeName(), extra, ("refs=" + str(refs)) if refs else "")
bb = cache.ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
print("TOTAL", bb)
