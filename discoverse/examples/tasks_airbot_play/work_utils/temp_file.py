#奇奇怪怪的临时文件/剪贴板

if stm.state_idx == 0:  # 伸到拱桥上方
    
    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
        [0.03, -0.015, 0.12]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 1:  # 伸到长方体上方
    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.12])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 3:  # 伸到长方体
    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.04])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 4:  # 抓住长方体
    sim_node.target_control[6] = 0.29
elif stm.state_idx == 5:  # 抓稳长方体
    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
elif stm.state_idx == 6:  # 提起长方体
    tmat_tgt_local[2, 3] += 0.09
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 7:  # 把长方体放到桥旁边上方
    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
        [0.075 + 0.00005, -0.015, 0.1]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 8:  # 保持夹爪角度 降低高度 把长方体放到桥旁边
    tmat_tgt_local[2, 3] -= 0.03
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 9:  # 松开方块
    sim_node.target_control[6] = 1
elif stm.state_idx == 10:  # 抬升高度
    tmat_tgt_local[2, 3] += 0.06
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )

elif stm.state_idx == 11:  # 伸到拱桥上方
    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
        [0.03, -0.015, 0.12]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 12:  # 伸到长方体上方
    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.12])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 14:  # 伸到长方体
    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.04])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 15:  # 抓住长方体
    sim_node.target_control[6] = 0.29
elif stm.state_idx == 16:  # 抓稳长方体
    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
elif stm.state_idx == 17:  # 提起长方体
    tmat_tgt_local[2, 3] += 0.09
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 18:  # 把长方体放到桥旁边上方
    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
        [-0.015 - 0.0005, -0.015, 0.1]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 19:  # 保持夹爪角度 降低高度 把长方体放到桥旁边
    tmat_tgt_local[2, 3] -= 0.03
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 20:  # 松开方块
    sim_node.target_control[6] = 1
elif stm.state_idx == 21:  # 抬升高度
    tmat_tgt_local[2, 3] += 0.06
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )

# 1
elif stm.state_idx == 22:  # 伸到立方体上方
    
    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 23:  # 伸到立方体
    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 24:  # 抓住立方体
    sim_node.target_control[6] = 0.24
elif stm.state_idx == 25:  # 抓稳立方体
    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
elif stm.state_idx == 26:  # 提起立方体
    tmat_tgt_local[2, 3] += 0.09
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 27:  # 把立方体放到长方体上方
    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
        [0, 0, 0.04 + 0.031 * 1]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 28:  # 把立方体放到长方体上侧
    tmat_tgt_local[2, 3] -= 0.01
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 29:  # 松开方块
    sim_node.target_control[6] = 1
elif stm.state_idx == 30:  # 抬升高度
    tmat_tgt_local[2, 3] += 0.02
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )

# 2
elif stm.state_idx == 31:  # 伸到立方体上方
    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple2")
    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
    sim_node.target_control[6] = 1
elif stm.state_idx == 32:  # 伸到立方体
    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple2")
    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 33:  # 抓住立方体
    sim_node.target_control[6] = 0.24
elif stm.state_idx == 34:  # 抓稳立方体
    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
elif stm.state_idx == 35:  # 提起立方体
    tmat_tgt_local[2, 3] += 0.09
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 36:  # 把立方体放到长方体上方
    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
        [0, 0, 0.04 + 0.031 * 2]
    )
    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 37:  # 把立方体放到长方体上侧
    tmat_tgt_local[2, 3] -= 0.01
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )
elif stm.state_idx == 38:  # 松开方块
    sim_node.target_control[6] = 1
elif stm.state_idx == 39:  # 抬升高度
    tmat_tgt_local[2, 3] += 0.02
    sim_node.target_control[:6] = arm_fik.properIK(
        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
    )