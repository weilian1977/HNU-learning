import matplotlib.pyplot as plt
import numpy as np
import time
import random
import warnings
import os


class _Monitor(object):

    def __init__(self, path) -> None:
        self.time_str: str = time.strftime("%Y-%m-%d-%H:%M:%S",
                                           time.localtime())
        self.path: str = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)


class _Line():

    def __init__(self, color) -> None:
        self.x_len: int = 0
        self.y: np.ndarray = np.zeros(shape=0, dtype=np.float32)
        self.color = color


class PlotMonitor(_Monitor):

    def __init__(self, save_path="./") -> None:
        super(PlotMonitor, self).__init__(save_path)
        self.lines: dict[str, _Line] = {}

    def add_line(self, line_label: str, color: str = None):
        c = color if color != None else (
            random.uniform(0.4, 1), random.uniform(0.4, 1),
            random.uniform(0.4, 1))  # Give a random RGB color
        self.lines[line_label] = _Line(color=c)

    def check_line_existence(self, line_label: str):
        return True if line_label in self.lines.keys() else False

    def add_data(self, line_label: str, new_value: float) -> None:
        try:
            self.lines[line_label].y = np.append(self.lines[line_label].y,
                                                 new_value)
            self.lines[line_label].x_len += 1
        except KeyError:
            warnings.warn(f"line \"{line_label}\" is not added")

    def generate_graph(self,
                       pic_name="1.jpg",
                       title: str = None,
                       xlabel: str = None,
                       ylabel: str = None) -> None:
        if len(self.lines) == 0:
            warnings.warn("No line has been added")
            return

        fig, axes = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
        for l in self.lines:
            x = np.linspace(0, 1, self.lines[l].x_len)
            axes.plot(x,
                      self.lines[l].y,
                      linestyle="solid",
                      color=self.lines[l].color,
                      marker='.',
                      label=l)
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.grid(axis='y', color='0.95')
        fig.legend(loc="upper left", title='Parameter where:')
        fig.suptitle(title)

        # save figure to file
        if os.path.splitext(pic_name)[1] not in (".jpg", ".png"):
            raise ValueError(
                f"Only support jpg or png picture, but receive {pic_name}")
        fig.savefig(os.path.join(self.path, pic_name),
                    facecolor='grey',
                    edgecolor='red')
        plt.close("all")

    def reset(self):
        self.x = np.zeros(shape=0, dtype=np.float32)
        self.y = np.zeros(shape=0, dtype=np.float32)
        self.cnt = 0


if __name__ == "__main__":
    import numpy as np

    p: PlotMonitor = PlotMonitor()

    p.add_line("acc")
    p.add_line("time", color="red")

    for i in range(50):
        p.add_data("acc", i)

    for i in range(120):
        p.add_data("time", np.sin(i / np.pi))

    p.generate_graph("asdf.jpg",
                     title="this is a title",
                     xlabel="time",
                     ylabel="hahha")
