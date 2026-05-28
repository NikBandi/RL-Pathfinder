import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots()

def plot(scores, mean_scores):
    ax.clear()
    ax.set_title('Training...')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    ax.plot(scores)
    ax.plot(mean_scores)
    ax.set_ylim(ymin=0)
    if scores:
        ax.text(len(scores)-1, scores[-1], str(scores[-1]))
        ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.01)
    fig.canvas.flush_events()