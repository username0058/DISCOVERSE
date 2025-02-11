class SimpleStateMachine:
    state_idx = 0      # 当前状态的索引
    state_cnt = 0      # 当前状态持续的计数
    new_state = True   # 标记是否是新状态
    max_state_cnt = -1 # 最大状态数，-1可能表示无限状态

    def next(self):
        if self.state_idx < self.max_state_cnt:
            self.state_cnt = 0    # 重置状态计数
            self.new_state = True # 标记为新状态
            self.state_idx += 1   # 状态索引加1

    def trigger(self):
        if self.new_state:
            self.new_state = False # 将新状态标记设为False
            return True            # 返回True表示触发了新状态
        else:
            return False           # 返回False表示不是新状态

    def update(self):
        self.state_cnt += 1  # 增加当前状态的计数

    def reset(self):
        self.state_idx = 0    # 重置状态索引
        self.state_cnt = 0    # 重置状态计数
        self.new_state = True # 重置为新状态

