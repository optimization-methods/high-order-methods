def polynomial_task(coefficients_number):
    def f(m_b, m_x):
        m_x = np.array(m_x, dtype=config.dtype)
        accumulator = 0
        for i in range(len(m_b)):
            accumulator += m_b[i] * m_x ** i
        return accumulator

    data = DatasetReader('planar').parse()
    xs = np.array(data.input)[:, 0]
    ys = data.output
    start = np.ones(coefficients_number)
    return Task(f, xs, ys, start)

def fractional_task(xs):
    def f(m_b, m_x):
        return m_b[0] * m_x / (m_b[1] + m_x)

    ys = f([2, 3], xs) + np.random.normal(0, 0.1, size=len(xs))
    start = [5, 5]
    return Task(f, xs, ys, start)

def polynomial_3d_task(x1, x2):
    def f(m_b, m_x):
        return m_b[0] - (1 / m_b[1]) * m_x[:, 0] ** 2 - (1 / m_b[2]) * m_x[:, 1] ** 2

    xs = np.column_stack([x1.ravel(), x2.ravel()])
    ys = f([4, 3, 2], xs) + np.random.normal(0, 1, size=len(xs))
    start = [1, 1, 1]
    return Task(f, xs, ys, start)