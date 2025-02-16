import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld
from mobile_sam import sam_model_registry, SamPredictor
from torch.cuda import Stream
import time
def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    xywh = [x1, y1, x2 - x1, y2 - y1]
    return tuple(xywh), list(box)

def render_frames(cam_num, output_queues, stop_event, init_event):
    """
    渲染窗口函数，实时显示处理后的图像
    :param cam_num: 摄像头数量
    :param output_queues: 输出队列列表，用于获取处理后的图像
    """
    # 创建窗口
    window_names = [f"Camera {i}" for i in range(cam_num)]
    # for name in window_names:
    #     cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    init_event.wait()  # 等待主线程准备好数据
    print("Rendering started. Press 'q' in any window to exit...")
    while not stop_event.is_set():
        # 从队列中读取图像
        frames = []
        for i in range(cam_num):
            if not output_queues[i].empty():
                frame = output_queues[i].get(block=False)
                frames.append(frame)
            else:
                frames.append(None)  # 如果没有新图像，保持上一帧

        # 显示图像
        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.imshow(window_names[i], frame)
        # 控制刷新频率
        key = cv2.waitKey(20)  # 40ms ≈ 25Hz，略高于20Hz以确保流畅
        if key == ord('q'):  # 按 'q' 键退出
            stop_event.set()
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()
  
class ObjectTracker:
    def __init__(self, cam_id, model_paths, class_names):
        """
        初始化目标跟踪器
        :param model_paths: 包含 YOLO 模型和 MobileSAM 模型路径的字典
        :param class_names: 目标类别名称列表
        """
        self.cam_id = cam_id
        self.class_names = class_names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化 YOLO 模型
        self.model_YL = YOLOWorld(model_paths["yolov8"])
        self.model_YL.set_classes(class_names)

        # 初始化 MobileSAM 模型
        self.sam_model = sam_model_registry["vit_t"](checkpoint=model_paths["mobile_sam"])
        self.sam_model.to(device=self.device)
        self.sam_model.eval()
        self.predictor = SamPredictor(self.sam_model)

        # 初始化 TrackerNano 参数
        self.tracker_params = cv2.TrackerNano_Params()
        self.tracker_params.backbone = model_paths["tracker_backbone"]
        self.tracker_params.neckhead = model_paths["tracker_head"]

        # 使用的标志位
        self.tracking = False
        self.show_video = False  # 是否显示视频
        self.track_count = 0
        self.max_low_mask = np.zeros((1, 256, 256), dtype=np.float32)
        self.output = None
        self.trackers = {}
        # 初始化CUDA流
        self.stream = Stream()

    def process_frame(self, frame):
        """
        处理每一帧图像
        :param frame: 当前帧图像
        :return: None
        """
        # 初始化变量
        failed_trackers = []
        dis_boxes = []
        all_masks_combined = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        self.predictor.set_image(frame)

        if not self.tracking:
            # YOLO-World 检测
            with torch.cuda.stream(self.stream):
                results = self.model_YL.predict(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            if boxes.shape[0] == 0:
            # 如果没有检测到目标，直接返回
                return
            probs = results[0].boxes.conf.cpu().numpy()
            names = results[0].boxes.cls.cpu().numpy().astype(int)


            # 提取指定类别的检测框
            for box, label, score in zip(boxes, names, probs):
                if label in self.class_names and score > 0:
                    dis_boxes.append(box.astype(int))
                    t_box, l_box = xyxy_to_xywh(box)
                    tracker = cv2.TrackerNano_create(self.tracker_params)
                    tracker.init(frame, t_box)
                    self.trackers[len(self.trackers)] = {
                        "tracker": tracker,
                        "name": self.class_names[label],
                        "box": l_box
                    }
            self.tracking = True
        else:
            # 使用 TrackerNano 进行目标跟踪
            for idx, data in self.trackers.items():
                tracker = data["tracker"]
                success, bbox = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in bbox]
                    dis_boxes.append([x, y, x + w, y + h])
                    data["box"] = [x, y, x + w, y + h]
                else:
                    failed_trackers.append(idx)
            self.track_count += 1
            if len(failed_trackers) > 0:
                self.tracking = False
                return
            if self.track_count > 39:
                self.track_count = 0
                self.tracking = False

        # 使用 MobileSAM 生成掩码
        with torch.cuda.stream(self.stream):
            if len(dis_boxes) > 0:
                for box in dis_boxes:
                    masks, iou_predictions, low_mask = self.predictor.predict(
                        box=np.array(box), 
                        multimask_output=True,
                        mask_input = self.max_low_mask
                        )
                max_iou_idx = np.argmax(iou_predictions)
                max_iou_mask = masks[max_iou_idx]
                pre_max_low_mask = low_mask[max_iou_idx]
                self.max_low_mask = pre_max_low_mask[None, :, :]
                all_masks_combined = np.maximum(all_masks_combined, max_iou_mask.astype(np.uint8) * 255)
                self.output = all_masks_combined
            else:
                self.output = None
        # 同步CUDA流，确保所有操作完成
        torch.cuda.synchronize(self.stream)
