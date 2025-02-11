import mujoco
import string
import numpy as np
import cv2
from mujoco import _enums
# This SimNode class is for type hinting only.
# The actual SimNode class is defined in another file.
class SimNode:
    pass

# def call_id(node:SimNode,target_body_name):
#     # 输入一个body名称，获取这个body名称对应的geom_id的所有列表
#     # 获取目标 body_id   
#     target_body_id = node.renderer_seg.model.body(target_body_name).id
#     # 获取所有 geom_id
#     geom_ids = []
#     print(node.renderer_seg.model.ngeom)

#     print(node.renderer_seg.model.geom_bodyid)
    
#     # 从 0 到 ngeom-1 的整数序列，即所有 geom 的 ID。 
#     for geom_id in range(node.renderer_seg.model.ngeom): # 0~89
#         # 这里认为geom_id与body_id是1对1的关系，即一个geom_id对应一个body_id，而且geom_id按0-90依次排列
#         if node.renderer_seg.model.geom_bodyid[geom_id] in [target_body_id]:# 拿出geomid对应的bodyid  10是block
#             geom_ids.append(geom_id)
#     print(geom_ids)
#     return geom_ids

def mask(node:SimNode) -> np.ndarray:#  
    if node.config.use_segmentation_renderer:
        if node.robot == "AirbotPlay":       
            node.renderer.update_scene(node.mj_data, node.camera_names[0], node.options)
            node.renderer_seg.update_scene(node.mj_data, node.camera_names[0], node.options)     
            node.renderer.scene.flags[_enums.mjtRndFlag.mjRND_SEGMENT] = True
            node.renderer.scene.flags[_enums.mjtRndFlag.mjRND_IDCOLOR] = True   
            # bowl_geoms_id = call_id(node,"block_green") 
            # bowl_geoms_id = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
            out = node.renderer.render()
            out[:] = np.flipud(out) # 消除正常的out翻转影响，注意此时输出的Out与真实方向相反
            image3 = out.astype(np.uint32)
            segimage = (
                image3[:, :, 0]
                + image3[:, :, 1] * (2**8)
                + image3[:, :, 2] * (2**16)
            )  # 求和得到指定的图片--只需要去查询对应的seg列表了
            node.renderer.scene.flags[_enums.mjtRndFlag.mjRND_SEGMENT] = False
            node.renderer.scene.flags[_enums.mjtRndFlag.mjRND_IDCOLOR] = False
            ###########################################################################################################################
            # seg = node.renderer_seg.render()
            # geom_ids = seg[:, :, 0]
            # geom_ids = geom_ids.astype(np.float64) + 1
            # geom_ids = geom_ids / geom_ids.max()
            # pixels = 255*geom_ids
            # seg_img = pixels.astype(np.uint8) #这里的seg_img是默认mujoco分割的输出结果
            # # out[:] = np.flipud(out) # 为了显示
            # iiii = out  # 实际上是_render.mjr_readPixels的输出
            # # cv2.imshow("out",out)
            # # cv2.imshow("out_seg",seg_img)
            # # cv2.waitKey(0)
            # # cv2.destroyAllWindows()
            scene = node.renderer_seg.scene
            ngeoms = scene.ngeom
            print(ngeoms)
            segid2output = np.full(
                (ngeoms, 1), fill_value = 0, dtype=np.int32
            )
            print(segid2output.shape)
            visible_geoms = [g for g in scene.geoms[:ngeoms] if g.segid != -1]
            # print(len(visible_geoms))
            visible_segids = np.array([g.segid for g in visible_geoms], np.int32)# 也就是geom_id
            # visible_objid = np.array([g.objid for g in visible_geoms], np.int32)
            # visible_objtype = np.array([g.objtype for g in visible_geoms], np.int32)
            print(visible_segids)
            print(range(node.renderer_seg.model.ngeom))
            # segid2output[visible_segids, 0] = 0 #其他物体都是0
            segid2output[bowl_geoms_id] = 255 # 碗是255
            # segid2output[visible_segids, 1] = visible_objtype
            out = segid2output[segimage]
            out[:] = np.flipud(out)
            out = out.squeeze()  # 移除多余的维度，从(448, 448, 1)变为(448, 448)
            out = out.astype(np.uint8)  # 确保数据类型为uint8
            print(out.shape)
            # binary_img = np.zeros_like(segimage, dtype=np.uint8)
            # # 将目标分割ID对应的区域设置为白色(255)
            # 创建一个布尔掩码，这个掩码的形状与 segimage 相同，对于 segimage 中值存在于 bowl_geoms_id 列表中的位置，对应的 mask 中的值为 True，否则为 False。
            # mask = np.isin(segimage, bowl_geoms_id)
            # 使用掩码来设置对应位置的值为 255
            # binary_img[mask] = 255
            # binary_img[:] = np.flipud(binary_img) # 为了显示         
            # geom_ids = out[:, :, 0]
            # geom_ids = geom_ids.astype(np.float64) + 1
            # geom_ids = geom_ids / geom_ids.max()
            # pixels = 255*geom_ids
            # seg_img_1 = pixels.astype(np.uint8)
            cv2.imshow("out",out)
            cv2.waitKey(0)  # 等待用户按键
            cv2.destroyAllWindows()

            # 将数据保存到文件
            with open('mask_output.txt', 'w') as f:
                 np.set_printoptions(threshold=np.inf)
                 f.write(f"visible_geoms: {visible_geoms}\n\n")
            #     f.write(f"visible_segids: {visible_segids.tolist()}\n\n")
            #     f.write(f"visible_objid: {visible_objid.tolist()}\n\n")
            #     f.write(f"visible_objtype: {visible_objtype.tolist()}\n")
                 f.write(f"segid2output:{segid2output}")
            #     np.set_printoptions(threshold=np.inf)
            #     f.write(f"out:{segimage}")
        elif node.robot == "mmk2":
            pass
        else:
            raise NotImplementedError("Other robots' mask is not implemented")
    else:
        pass
