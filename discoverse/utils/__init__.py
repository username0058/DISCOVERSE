from .controllor import PIDController, PIDarray
from .base_config import BaseConfig
from .statemachine import SimpleStateMachine

import numpy as np
from scipy.spatial.transform import Rotation
#TODO：仿真器里面根本没有使用PID控制器而只是采用了step的方式
def get_site_tmat(mj_data, site_name):
    tmat = np.eye(4)
    tmat[:3,:3] = mj_data.site(site_name).xmat.reshape((3,3))
    tmat[:3,3] = mj_data.site(site_name).xpos
    return tmat

def get_body_tmat(mj_data, body_name):
    tmat = np.eye(4)
    tmat[:3,:3] = Rotation.from_quat(mj_data.body(body_name).xquat[[1,2,3,0]]).as_matrix()
    tmat[:3,3] = mj_data.body(body_name).xpos
    return tmat
# 这个函数 step_func 实现了一个简单的平滑移动算法。它的目的是将一个当前值（current）逐步调整到目标值（target），每次调整的幅度不超过给定的步长（step）。让我们逐行分析：

# if current < target - step:

# 如果当前值小实现了一个简单的平滑移动算法。它的目的是将一个当前值（current）逐步调整到目标值（target），每次调整的幅度不超过给定的步长（step）。让我们逐行分析：于目标值减去步长，说明当前值还远低于目标值。
# 在这种情况下，函数返回 current + step，即将当前值增加一个步长。
# elif current > target + step:

# 如果当前值大于目标值加上步长，说明当前值还远高于目标值。
# 在这种情况下，函数返回 current - step，即将当前值减少一个步长。
# else:

# 如果当前值在目标值的步长范围内（即 target - step <= current <= target + step），
# 函数直接返回目标值 target。
# 这个函数的作用是：

# 实现平滑移动：不是直接跳到目标值，而是逐步接近。
# 限制每次移动的幅度：每次移动不会超过给定的步长。
# 确保最终达到目标：当接近目标值时，直接设置为目标值，避免在目标值附近震荡。
# 这种方法在机器人控制中很有用，可以用于：

# 平滑关节运动，避免突然的大幅度变化。
# 实现速度限制，防止关节移动过快。
# 在接近目标位置时提供精确定位。
# 总的来说，这个函数提供了一种简单而有效的方法来实现平滑、受控的值调整，这在机器人运动控制、动画、或任何需要平滑过渡的场景中都非常有用。
def step_func(current, target, step):
    if current < target - step:
        return current + step
    elif current > target + step:
        return current - step
    else:
        return target
