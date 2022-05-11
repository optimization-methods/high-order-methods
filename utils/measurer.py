class Measurer:
    def __init__(self, descent_results):
        self.descent_results = descent_results

    def draw_time(self):
        for descent_result in self.descent_results:
            descent_result.pigify()

    def draw_memory(self):
        for descent_result in self.descent_results:
            descent_result.pigify()
