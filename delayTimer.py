import time
from threading import Thread, Lock

class FixedTimeBuffer:
    def __init__(self, pop_interval, exec_callback):
        """
        初始化定时pop Buffer。
        :param pop_interval: 每个数据在Buffer中的固定停留时间（单位：秒）。
        :param exec_callback: 当数据被弹出时执行的回调函数，接受弹出数据作为参数。
        """
        self.buffer = []  # 存储 (timestamp, data) 的列表
        self.pop_interval = pop_interval
        self.exec_callback = exec_callback
        self.lock = Lock()  # 确保线程安全
        self.running = True
        self.pop_thread = Thread(target=self._pop_loop)
        self.pop_thread.daemon = True
        self.pop_thread.start()

    def add(self, data):
        """
        向Buffer中添加数据。
        :param data: 要添加的数据。
        """
        with self.lock:
            timestamp = time.time()
            self.buffer.append((timestamp, data))

    def _pop_loop(self):
        """
        定时pop数据的后台线程。
        """
        while self.running:
            time.sleep(0.01) 
            self._pop_expired()

    def _pop_expired(self):
        """
        移除超过pop_interval的数据，并对其进行处理。
        """
        with self.lock:
            current_time = time.time()
            while self.buffer and current_time - self.buffer[0][0] >= self.pop_interval:
                timestamp, data = self.buffer.pop(0)
                self.handle_pop(data)

    def handle_pop(self, data):
        """
        处理被pop的数据。
        :param data: 被pop的数据。
        """
        # 调用回调函数
        self.exec_callback(data)

    def stop(self):
        """
        停止后台线程。
        """
        self.running = False
        self.pop_thread.join()

# 示例用法
if __name__ == "__main__":
    # 初始化Buffer，每个数据在Buffer中停留3秒后pop
    buffer = FixedTimeBuffer(pop_interval=0.01)

    # 模拟向Buffer中添加数据
    for i in range(5):
        time.sleep(0.1)  # 每秒添加一个数据
        buffer.add(f"action_{i}")
        print(f"Added: action_{i}")

    # 主程序等待10秒后停止
    # time.sleep(10)
    buffer.stop()


# import time
# from threading import Thread, Lock

# class FixedTimeBuffer:
#     def __init__(self, pop_interval, exec_callback=None, max_size=10):
#         """
#         初始化定时pop Buffer。
#         :param pop_interval: 每个数据在Buffer中的固定停留时间（单位：秒）。
#         :param exec_callback: 当数据被弹出时执行的回调函数，接受弹出数据作为参数。
#         :param max_size: Buffer中允许的最大数据数量。
#         """
#         self.buffer = []  # 存储 (timestamp, data) 的列表
#         self.pop_interval = pop_interval
#         self.exec_callback = exec_callback
#         self.max_size = max_size
#         self.lock = Lock()  # 确保线程安全
#         self.running = True
#         self.pop_thread = Thread(target=self._pop_loop)
#         self.pop_thread.daemon = True
#         self.pop_thread.start()

#     def add(self, data):
#         """
#         向Buffer中添加数据，如果超过max_size限制则丢弃最早的数据。
#         :param data: 要添加的数据。
#         """
#         with self.lock:
#             timestamp = time.time()
#             if len(self.buffer) >= self.max_size:
#                 # 丢弃最早的数据
#                 oldest = self.buffer.pop(0)
#                 print(f"Buffer full. Dropping oldest data: {oldest}")
#             self.buffer.append((timestamp, data))
#             print(f"Added to buffer: {data} (current size: {len(self.buffer)})")

#     def _pop_loop(self):
#         """
#         定时pop数据的后台线程。
#         """
#         while self.running:
#             time.sleep(0.001)  # 每毫秒检查一次
#             self._pop_expired()

#     def _pop_expired(self):
#         """
#         移除超过pop_interval的数据，并对其进行处理。
#         """
#         with self.lock:
#             current_time = time.time()
#             while self.buffer and current_time - self.buffer[0][0] >= self.pop_interval:
#                 timestamp, data = self.buffer.pop(0)
#                 self.handle_pop(data)

#     def handle_pop(self, data):
#         """
#         处理被pop的数据。
#         :param data: 被pop的数据。
#         """
#         if self.exec_callback:
#             self.exec_callback(data)
#         else:
#             print(f"Popped: {data}")

#     def stop(self):
#         """
#         停止后台线程。
#         """
#         self.running = False
#         self.pop_thread.join()
