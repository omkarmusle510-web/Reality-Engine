from pathlib import Path

from apps.reality_painter.assets.retriever import AssetRetriever
from apps.reality_painter.assets.github import discover_assets
from engine.scene.loader import load_glb
from engine.scene.scene import Scene
from engine.rendering.renderer import Renderer3D

assets = discover_assets("KhronosGroup/glTF-Sample-Assets")
asset = next(a for a in assets if a.format == "glb")

retriever = AssetRetriever(cache_dir=Path("assets_cache"))
path = retriever.retrieve(asset)

print("GLB:", path)
print("Exists:", path.exists())
print("Size:", path.stat().st_size)

print("Loading GLB...")
obj = load_glb(path, name=asset.id)

print("Loaded:", obj.name)

scene = Scene()
scene.add(obj)

print("Scene objects:", len(scene.objects()))

print("Rendering...")
renderer = Renderer3D(width=640, height=480)

image = renderer.render(scene)

print("Rendered:", image.shape)
print("dtype:", image.dtype)
print("Non-zero pixels:", int((image != 0).sum()))

renderer.close()

print("REAL 12D TEST PASSED")