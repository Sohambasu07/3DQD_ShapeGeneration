import open3d as o3d    
import trimesh

path = '/Users/yusufsalcan/Documents/ESE_Semester_2/Deep_Learning_Lab/project/ShapeNetCore.v2/02691156/1caa02b831cccff090baeef8ba5b93e5/models/model_normalized.obj'
mesh_in = o3d.io.read_triangle_mesh(path)
mesh_in.compute_vertex_normals()

print("Before Simplification: ", mesh_in)
# o3d.visualization.draw_geometries([mesh_in])

mesh_smp = mesh_in.simplify_quadric_decimation(
    target_number_of_triangles=5000)
print("After Simplification target number of triangles = 5000:\n", mesh_smp)
# o3d.visualization.draw_geometries([mesh_smp])
print(type(mesh_smp))
print(mesh_smp.vertices)
vertices = mesh_smp.vertices
triangles = mesh_smp.triangles
mesh = trimesh.Trimesh(vertices, triangles)
mesh.show()

# mesh = trimesh.load_mesh(path)
# meshes = mesh.dump(concatenate=True)
# merged_mesh = trimesh.util.concatenate(meshes)
# merged_mesh.show()
