# FILEPATH: /home/djr/DISCOVERSE/discoverse/examples/tasks_use_mask/block_place_multi.py

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from vison_seg import ObjectTracker
import torch
import os
from datetime import datetime

class ImageProcessor:
    def __init__(self, cam_num, model_paths, class_names, display=True):
        self.cam_num = cam_num
        self.model_paths = model_paths
        self.class_names = class_names
        self.bridge = CvBridge()
        self.trackers = [ObjectTracker(i, model_paths, class_names) for i in range(cam_num)]
        self.display = display

        # 创建订阅者
        self.subscribers = [
            rospy.Subscriber(f'camera_{i}/image', Image, self.image_callback, callback_args=i)
            for i in range(cam_num)
        ]

        # 创建发布者
        self.publishers = [
            rospy.Publisher(f'processed_camera_{i}/image', Image, queue_size=10)
            for i in range(cam_num)
        ]

        if self.display:
            # 创建显示窗口
            for i in range(cam_num):
                cv2.namedWindow(f"Processed Camera {i}", cv2.WINDOW_NORMAL)
        
        # 记录最后一次接收到消息的时间
        self.last_message_time = rospy.Time.now()
    def image_callback(self, msg, camera_id):
        # 更新最后一次接收到消息的时间
        self.last_message_time = rospy.Time.now()
        # 将 ROS 图像消息转换为 NumPy 数组
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        # 处理图像
        self.trackers[camera_id].process_frame(frame)
        processed_frame = self.trackers[camera_id].output

        if processed_frame is not None:
            # 将处理后的图像转换回 ROS 消息并发布
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="mono8")
            processed_msg.header = msg.header  # 保持原始消息的时间戳和帧 ID
            self.publishers[camera_id].publish(processed_msg)

            if self.display:
                # 显示处理后的图像
                # TODO：根本不显示！！！！！！！！！！！！！！！(权限问题)
                print("Displaying.......")
                # cv2.imshow(f"Processed Camera {camera_id}", processed_frame)
                # cv2.waitKey(1)  # 允许窗口刷新
                # 保存处理后的图像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"camera_{camera_id}_{timestamp}.jpg"
                filepath = os.path.join("/home/djr/DISCOVERSE/discoverse/output_images/", filename)
                if not cv2.imwrite(filepath, processed_frame):
                    print(f"Failed to save image to {filepath}")
                else:
                    print(f"Successfully saved image to {filepath}")
                print("cv2")
            
        else:
            rospy.logwarn(f"Camera {camera_id} produced no output for this frame.")

    def check_message_timeout(self):
        # 检查是否超过一定时间未接收到消息
        current_time = rospy.Time.now()
        timeout_duration = rospy.Duration(2)  # 2 秒超时时间
        if (current_time - self.last_message_time) > timeout_duration:
            rospy.logerr("No message received for 2 seconds. Shutting down node.")
            self.shutdown()

    def shutdown(self):
        if self.display:
            cv2.destroyAllWindows()
        rospy.signal_shutdown("No message received. Shutting down.")
        # 在整个过程结束时显式地释放资源
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



def main():
    torch.cuda.empty_cache()
    rospy.init_node('image_processor', anonymous=True)

    cam_num = 2  # 假设有两个相机
    model_paths = {
        "yolov8": "/home/djr/MobileSAM/weights/yolov8x-world.pt",
        "mobile_sam": "/home/djr/MobileSAM/weights/mobile_sam.pt",
        "tracker_backbone": "/home/djr/MobileSAM/weights/nanotrack_backbone_sim.onnx",
        "tracker_head": "/home/djr/MobileSAM/weights/nanotrack_head_sim.onnx"
    }
    class_names = ["block", "bowl"]

    image_processor = ImageProcessor(cam_num, model_paths, class_names, display=True)

    rate = rospy.Rate(10)  # 10Hz，检查频率
    while not rospy.is_shutdown():
        # 每次循环检查超时
        image_processor.check_message_timeout()
        # 使用spin_once代替spin来处理一次回调
        rospy.spin_once()
        rate.sleep()  # 控制循环频率
    
if __name__ == "__main__":
    main()
