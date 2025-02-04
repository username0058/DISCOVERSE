import os
import time
import traceback
from abc import abstractmethod
from multiprocessing import Process, shared_memory, Value, Array

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.utils import BaseConfig

try:
    from discoverse.gaussian_renderer import GSRenderer
    from discoverse.gaussian_renderer.util_gau import multiple_quaternion_vector3d, multiple_quaternions
    DISCOVERSE_GAUSSIAN_RENDERER = True

except ImportError:
    traceback.print_exc()
    print("Warning: gaussian_splatting renderer not found. Please install the required packages to use it.")
    DISCOVERSE_GAUSSIAN_RENDERER = False


def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

def imshow_loop(render_cfg, shm, key, mouseParam):
    def mouseCallback(event, x, y, flags, param):
        mouseParam[0] = event
        mouseParam[1] = x
        mouseParam[2] = y
        mouseParam[3] = flags

    cv_windowname = render_cfg["cv_windowname"]
    cv2.namedWindow(cv_windowname, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(cv_windowname, render_cfg["width"], render_cfg["height"])
    cv2.setMouseCallback(cv_windowname, mouseCallback)

    img_vis_shared = np.ndarray((render_cfg["height"], render_cfg["width"], 3), dtype=np.uint8, buffer=shm.buf)

    set_fps = min(render_cfg["fps"], 60.)
    time_delay = 1./set_fps
    time_delay_ms = int(time_delay * 1e3 - 1.0)
    while cv2.getWindowProperty(cv_windowname, cv2.WND_PROP_VISIBLE):
        t0 = time.time()
        cv2.imshow(cv_windowname, img_vis_shared)
        key.value = cv2.waitKey(time_delay_ms)
        t1 = time.time()
        time.sleep(max(time_delay - (t1-t0), 0.0))
    key.value = -2
    cv2.destroyAllWindows()
    print("imshow_loop is terminated")

    time.sleep(0.1)
    shm.close()
    shm.unlink()
    shm = None

class SimulatorBase:
    running = True
    obs = None

    cam_id = -1
    last_cam_id = -1
    render_cnt = 0
    camera_names = []
    mouse_last_x = 0
    mouse_last_y = 0

    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    options = mujoco.MjvOption()

    def __init__(self, config:BaseConfig):
        # 保存配置对象
        self.config = config

        # 确定MJCF文件的路径
        if self.config.mjcf_file_path.startswith("/"):
            self.mjcf_file = self.config.mjcf_file_path
        else:
            self.mjcf_file = os.path.join(DISCOVERSE_ASSERT_DIR, self.config.mjcf_file_path)
        
        # 检查MJCF文件是否存在
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))
        
        # 初始化自由相机
        self.free_camera = mujoco.MjvCamera()# 创建一个MjvCamera对象作为自由相机
        self.free_camera.fixedcamid = -1
        self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE# 设置相机类型为自由相机

        # 加载MJCF文件并设置默认自由相机
        self.load_mjcf()
        mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)# 设置默认的自由相机参数

        # 配置高斯渲染器（如果启用）
        self.config.use_gaussian_renderer = self.config.use_gaussian_renderer and DISCOVERSE_GAUSSIAN_RENDERER
        if self.config.use_gaussian_renderer:
            self.gs_renderer = GSRenderer(self.config.gs_model_dict, self.config.render_set["width"], self.config.render_set["height"])
            self.last_cam_id = self.cam_id
            self.show_gaussian_img = True
            # 设置相机的视场角（FOV）
            if self.cam_id == -1:
                self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.)
            else:
                self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[self.cam_id] * np.pi / 180.0)

        # 设置模拟参数
        # 计算模拟时间步长：
        # 基于模型的时间步长和抽取因子（decimation）计算
        self.decimation = self.config.decimation
        self.delta_t = self.mj_model.opt.timestep * self.decimation
        self.render_fps = self.config.render_set["fps"]

        # 检查对象名称（如果使用高斯渲染器）
        if self.config.use_gaussian_renderer:
            obj_names_check = True
            obj_names = self.mj_model.names.decode().split("\x00")
            for name in self.config.rb_link_list + self.config.obj_list:
                if not name in obj_names:
                    print(f"\033[0;31;40mInvalid object name: {name}\033[0m")
                    obj_names_check = False
            assert obj_names_check, "ERROR: Invalid object name"

        # 设置默认MuJoCo选项
        mujoco.mjv_defaultOption(self.options)

        # 设置非无头模式下的渲染参数
        if not self.config.headless:
            self.config.render_set["cv_windowname"] = self.mj_model.names.decode().split("\x00")[0].upper()

            # 创建共享内存用于图像数据
            self.shm = shared_memory.SharedMemory(create=True, size=(self.config.render_set["height"] * self.config.render_set["width"] * 3) * np.uint8().itemsize)
            self.img_vis_shared = np.ndarray((self.config.render_set["height"], self.config.render_set["width"], 3), dtype=np.uint8, buffer=self.shm.buf)
            self.key = Value('i', lock=True)
            self.mouseParam = Array("i", 4, lock=True)

            # 启动图像显示进程
            self.imshow_process = Process(target=imshow_loop, args=(self.config.render_set, self.shm, self.key, self.mouseParam))
            self.imshow_process.start()

        # 初始化渲染时间和模型数据
        # 重置MuJoCo数据并执行初始的前向动力学计算：
        # 记录最后渲染时间。
        # 重置模型数据。
        # 执行前向动力学计算
        self.last_render_time = time.time()
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)


    def load_mjcf(self):
        if self.mjcf_file.endswith(".xml"):
            self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        elif self.mjcf_file.endswith(".mjb"):
            self.mj_model = mujoco.MjModel.from_binary_path(self.mjcf_file)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)
        for i in range(self.mj_model.ncam):
            self.camera_names.append(self.mj_model.camera(i).name)

        if type(self.config.obs_rgb_cam_id) is int:
            assert -2 < self.config.obs_rgb_cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(self.config.obs_rgb_cam_id)
            tmp_id = self.config.obs_rgb_cam_id
            self.config.obs_rgb_cam_id = [tmp_id]
        elif type(self.config.obs_rgb_cam_id) is list:
            for cam_id in self.config.obs_rgb_cam_id:
                assert -2 < cam_id < len(self.camera_names), "Invalid obs_rgb_cam_id {}".format(cam_id)
        elif self.config.obs_rgb_cam_id is None:
            self.config.obs_rgb_cam_id = []
        
        if type(self.config.obs_depth_cam_id) is int:
            assert -2 < self.config.obs_depth_cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(self.config.obs_depth_cam_id)
        elif type(self.config.obs_depth_cam_id) is list:
            for cam_id in self.config.obs_depth_cam_id:
                assert -2 < cam_id < len(self.camera_names), "Invalid obs_depth_cam_id {}".format(cam_id)
        elif self.config.obs_depth_cam_id is None:
            self.config.obs_depth_cam_id = []

        self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])

        self.post_load_mjcf()

    def post_load_mjcf(self):
        pass

    def __del__(self):
        if not self.config.headless:
            self.imshow_process.join()
            self.shm.close()
            self.shm.unlink()
        try:
            self.renderer.close()
        except AttributeError as ae:
            pass
        finally:
            print("SimulatorBase is deleted")

    def cv2MouseCallback(self):
        event = self.mouseParam[0]
        x = self.mouseParam[1]
        y = self.mouseParam[2]
        flags = self.mouseParam[3]

        self.mouseParam[0] = 0
        self.mouseParam[3] = 0

        if self.cam_id == -1:
            action = None
            if flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif flags == cv2.EVENT_FLAG_RBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif flags == cv2.EVENT_FLAG_MBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            if not action is None:
                self.camera_pose_changed = True
                height = self.config.render_set["height"]
                dx = float(x) - self.mouse_last_x
                dy = float(y) - self.mouse_last_y
                mujoco.mjv_moveCamera(self.mj_model, action, dx/height, dy/height, self.renderer.scene, self.free_camera)
        self.mouse_last_x = float(x)
        self.mouse_last_y = float(y)

    def update_gs_scene(self):
        for name in self.config.obj_list + self.config.rb_link_list:
            trans, quat_wxyz = self.getObjPose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def getRgbImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
                self.gs_renderer.set_camera_fovy(self.mj_model.vis.global_.fovy * np.pi / 180.0)
            if self.last_cam_id != cam_id and cam_id > -1:
                self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render()
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            rgb_img = self.renderer.render()
            return rgb_img

    def getDepthImg(self, cam_id):
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            if self.last_cam_id != cam_id:
                if cam_id == -1:
                    self.gs_renderer.set_camera_fovy(np.pi * 0.25)
                elif cam_id > -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
                else:
                    return None
            self.last_cam_id = cam_id
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render(render_depth=True)
        else:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            elif cam_id > -1:
                self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
            else:
                return None
            depth_img = self.renderer.render()
            return depth_img

    def cv2WindowKeyPressCallback(self, key):
        if key == -1:
            return True
        elif key == -2:
            return False
        elif key == ord('h'):
            self.printHelp()
        elif key == ord("p"):
            self.printMessage()
        elif key == ord('r'):
            self.reset()
        elif key == 194: #F5
            self.renderer.close()
            self.load_mjcf()
            self.reset()
        elif key == ord('g') and self.config.use_gaussian_renderer:
            self.show_gaussian_img = not self.show_gaussian_img
            self.gs_renderer.renderer.need_rerender = True
        elif key == ord('d'):
            if self.config.use_gaussian_renderer:
                self.gs_renderer.renderer.need_rerender = True
            if self.renderer._depth_rendering:
                self.renderer.disable_depth_rendering()
            else:
                self.renderer.enable_depth_rendering()
        elif key == 27: # "ESC"
            self.cam_id = -1
            self.camera_pose_changed = True
        elif key == ord(']') and self.mj_model.ncam:
            self.cam_id += 1
            self.cam_id = self.cam_id % self.mj_model.ncam
        elif key == ord('[') and self.mj_model.ncam:
            self.cam_id += self.mj_model.ncam - 1
            self.cam_id = self.cam_id % self.mj_model.ncam
        return True
    
    def printHelp(self):
        print("Press 'h' to print help")
        print("Press 'r' to reset the state")
        print("Press '[' or ']' to switch camera view")
        print("Press 'Esc' to set free camera")
        print("Press 'p' to print the rotot state")
        print("Press 'g' toggle gaussian render")
        print("Press 'd' toggle depth render")

    def printMessage(self):
        pass

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.camera_pose_changed = True

    def getCameraPose(self, cam_id):
        """
        获取指定相机的位置和方向。

        参数:
        cam_id (int): 相机ID。-1表示自由相机，其他值表示预定义相机。

        返回:
        tuple: 包含两个元素：
            - camera_position (numpy.ndarray): 相机的3D位置。
            - camera_orientation (numpy.ndarray): 相机的方向，以四元数形式表示。
        """
        if cam_id == -1:
            # 处理自由相机
            # 计算旋转矩阵，考虑相机的仰角和方位角
            rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            # 计算相机位置，基于注视点、距离和旋转矩阵
            camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        else:
            # 处理预定义相机
            # 从MuJoCo数据中直接获取旋转矩阵和位置
            rotation_matrix = np.array(self.mj_data.camera(self.camera_names[cam_id]).xmat).reshape((3,3))
            camera_position = self.mj_data.camera(self.camera_names[cam_id]).xpos

        # 返回相机位置和方向（四元数形式，调整顺序）
        return camera_position, Rotation.from_matrix(rotation_matrix).as_quat()[[3,0,1,2]]


    def getObjPose(self, name):
        """
        获取指定对象的位置和方向。

        参数:
        name (str): 对象的名称。

        返回:
        tuple: 包含两个元素：
            - position (numpy.ndarray): 对象的3D位置。
            - quat (numpy.ndarray): 对象的方向，以四元数形式表示。
        如果对象不存在，则返回 (None, None)。
        """
        try:
            # 尝试获取body对象的位置和方向
            position = self.mj_data.body(name).xpos
            quat = self.mj_data.body(name).xquat
            return position, quat
        except KeyError:
            try:
                # 如果不是body，尝试获取geom对象的位置和方向
                position = self.mj_data.geom(name).xpos
                # 将旋转矩阵转换为四元数，并调整四元数的顺序
                quat = Rotation.from_matrix(self.mj_data.geom(name).xmat.reshape((3,3))).as_quat()[[3,0,1,2]]
                return position, quat
            except KeyError:
                # 如果既不是body也不是geom，打印错误信息并返回None
                print("Invalid object name: {}".format(name))
                return None, None


    def render(self):
        """
        渲染当前场景，更新观察图像，并处理用户交互。

        此方法负责更新场景状态，生成RGB和深度图像，处理用户输入，
        并根据配置同步渲染过程。
        """
        self.render_cnt += 1

        # 如果启用了高斯渲染器并需要显示高斯图像，则更新高斯场景
        if self.config.use_gaussian_renderer and self.show_gaussian_img:
            self.update_gs_scene()

        # 生成RGB观察图像
        self.img_rgb_obs_s = {}
        for id in self.config.obs_rgb_cam_id:
            img = self.getRgbImg(id)
            self.img_rgb_obs_s[id] = img

        # 生成深度观察图像
        self.img_depth_obs_s = {}
        for id in self.config.obs_depth_cam_id:
            img = self.getDepthImg(id)
            self.img_depth_obs_s[id] = img

        # 准备可视化图像
        if not self.renderer._depth_rendering:
            # 如果不是深度渲染，准备RGB图像
            if self.cam_id in self.config.obs_rgb_cam_id:
                img_vis = cv2.cvtColor(self.img_rgb_obs_s[self.cam_id], cv2.COLOR_RGB2BGR)
            else:
                img_rgb = self.getRgbImg(self.cam_id)
                img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            # 如果是深度渲染，准备深度图像
            if self.cam_id in self.config.obs_depth_cam_id:
                img_depth = self.img_depth_obs_s[self.cam_id]
            else:
                img_depth = self.getDepthImg(self.cam_id)
            if not img_depth is None:
                img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=25.5), cv2.COLORMAP_JET)

        # 如果不是无头模式，将可视化图像复制到共享内存
        if not self.config.headless:
            np.copyto(self.img_vis_shared, img_vis)

        # 如果启用同步，等待以保持指定的渲染帧率
        if self.config.sync:
            wait_time_s = max(1./self.render_fps - time.time() + self.last_render_time, 0.0)
            time.sleep(wait_time_s)

        # 如果不是无头模式，处理用户输入
        if not self.config.headless:
            self.cv2MouseCallback()
            if not self.cv2WindowKeyPressCallback(self.key.value):
                self.running = False
            self.key.value = -1

        # 更新上次渲染时间
        self.last_render_time = time.time()


    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateControl(self, action):
        pass

    @abstractmethod
    def post_physics_step(self):
        pass

    @abstractmethod
    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    @abstractmethod
    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")    

    @abstractmethod
    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    @abstractmethod
    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    @abstractmethod
    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------
    # 接受一步action，做一步/多步物理仿真，返回observation, reward, done, info
    def step(self, action=None):
        for _ in range(self.decimation):
            self.updateControl(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.checkTerminated():
            self.resetState()
        
        self.post_physics_step()
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()

        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), self.checkTerminated(), {}

    def view(self):
        # 更新模拟时间
        self.mj_data.time += self.delta_t

        # 重置所有关节速度为零
        self.mj_data.qvel[:] = 0

        # 调用MuJoCo的前向动力学函数，更新模拟状态
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 检查是否需要渲染新帧
        # 这确保了渲染以指定的帧率进行，不受模拟步骤频率的影响
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            # 如果是时候渲染新帧，调用渲染方法
            self.render()

