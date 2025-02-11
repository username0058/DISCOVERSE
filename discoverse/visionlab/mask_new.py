import mujoco
import string
import numpy as np
import cv2
import mediapy as media
# This SimNode class is for type hinting only.
# The actual SimNode class is defined in another file.
class SimNode:
    pass

def mask(node:SimNode,target_body_name) -> np.ndarray:#  
    if node.config.use_segmentation_renderer:   
        node.renderer_seg.update_scene(node.mj_data, node.camera_names[0], node.options)     
        seg = node.renderer_seg.render()
        geom_ids_ori = seg[:, :, 0]
        body_name = target_body_name
        # print(node.renderer_seg.model.body(body_name).geomadr, node.renderer_seg.model.body(body_name).geomnum)
        mask = np.zeros_like(geom_ids_ori, dtype=np.uint8)
        # print(node.renderer_seg.model.body(body_name).geomadr, node.renderer_seg.model.body(body_name).geomnum)
        mask[np.where((node.renderer_seg.model.body(body_name).geomadr <= geom_ids_ori) & (geom_ids_ori < node.renderer_seg.model.body(body_name).geomadr + node.renderer_seg.model.body(body_name).geomnum))] = 255
        # media.show_image(mask)    
        # cv2.imshow("mask",mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return mask
    else:
        pass

