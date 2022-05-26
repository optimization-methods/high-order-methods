from dataclasses import dataclass



# class PolynomialTaskType(TaskType):
#     __init__ = super().__init__
#
#     def f(self, m_b, m_x):
#         m_x = np.array(m_x, dtype=config.dtype)
#         accumulator = 0
#         for i in range(len(m_b)):
#             accumulator += m_b[i] * m_x ** i
#         return accumulator
#
# class MovingTaskType(TaskType):
#     __init__ = super().__init__
#
#     def f(self, m_b, m_x):
#         return m_b[0] * m_x / (m_b[1] + m_x)
#
# class DatasetTaskType(TaskType):
#     __init__ = super().__init__
#
#     def f(self, m_b, m_x):
#         return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2
