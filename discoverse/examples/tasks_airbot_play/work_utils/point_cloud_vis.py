import os
import open3d as o3d

def visualize_ply_files(directory='point_cloud'):
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # 获取所有 .ply 文件
    ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]
    
    if not ply_files:
        print("No .ply files found in the directory.")
        return
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # 读取并可视化所有 .ply 文件
    for ply_file in ply_files:
        file_path = os.path.join(directory, ply_file)
        print(f"Loading: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name=ply_file)

if __name__ == "__main__":
    visualize_ply_files()