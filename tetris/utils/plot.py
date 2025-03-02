import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme()


class ScatterPlot:
    margin = 0.05

    def __init__(self, xlabel, ylabel, title, log_scale=False) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x, self.y = [], []
        self.sc = self.ax.scatter(self.x, self.y)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.log_scale = log_scale
        plt.draw()

    def add_point(self, x, y, render, pause=0.0001):
        self.x.append(x)
        self.y.append(y)
        if not render:
            return
        self.update(pause)

    def toggle_log_scale(self):
        self.log_scale = not self.log_scale
        self.set_scale()
        self.update()

    def update(self, pause=0.0001):
        if len(self.x) < 2:
            return
        ax = plt.gca()
        xmin, xmax = min(self.x), max(self.x)
        ymin, ymax = min(self.y), max(self.y)
        dx, dy = (xmax - xmin) * self.margin, (ymax - ymin) * self.margin
        ax.set_xlim([xmin - dx, xmax + dx])
        ax.set_ylim([ymin - dy, ymax + dy])
        self.sc.set_offsets(np.c_[self.x, self.y])
        self.fig.canvas.draw_idle()
        plt.pause(pause)

    def set_scale(self):
        # self.ax.set_xscale("log" if self.log_scale else "linear")
        self.ax.set_yscale("log" if self.log_scale else "linear")

    def freeze(self):
        plt.waitforbuttonpress()

    def stats(self):
        if len(self.x) < 1:
            return 0, 0, 0
        y = np.array(self.y)
        return np.round(np.mean(y), 2), np.round(np.std(y), 2), np.max(y)

    def save(self, name):
        np.savetxt(name, self.y, delimiter=",")


if __name__ == "__main__":
    data = pd.read_csv("genetic_scores.csv")[:3000]
    data.reset_index(inplace=True)
    data.columns = ["Index", "Score"]

    # Plotting the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Index", y="Score", data=data, edgecolor="none")
    plt.title("Score per game while training")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.show()
