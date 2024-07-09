import bpy
import os 

mainfile_path = "sample.blend"

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.context.scene.cursor.location = (0, 0, 0)

filepath = os.path.join(os.path.dirname(__file__), "target.png")

bpy.ops.wm.ply_import(filepath="test.ply")


bpy.ops.wm.save_mainfile(filepath=mainfile_path)
os.system('blender ./sample.blend')
# bpy.ops.wm.open_mainfile(filepath = mainfile_path)
# # 拿到当前的场景
# scene = bpy.context.scene 
# # 设定渲染的目标地址
# scene.render.filepath = filepath
# # 渲染
# bpy.ops.render.render(write_still=True)
# # 保存文件，相当于 ctrl + s
# bpy.ops.wm.save_mainfile(filepath=mainfile_path)

