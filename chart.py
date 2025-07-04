import json
import matplotlib.pyplot as plt

def plot_episode_data(json_path, figure_name='episode_data_plot.png'):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Ekstrak data per episode
    episodes = [d["episode"] for d in data]
    scores = [d["score"] for d in data]
    mean_scores = [d["mean_score"] for d in data]
    epsilons = [d["epsilon"] for d in data]

    # Buat scatter plot
    plt.figure(figsize=(14, 8))

    # Plot 1: Score dan Mean Score
    plt.subplot(1, 1, 1)
    plt.scatter(episodes, scores, s=10, alpha=0.6, label='Score')
    plt.scatter(episodes, mean_scores, s=10, alpha=0.6, label='Mean Score')
    plt.title('Scores per Episode (Scatter)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # # Plot 2: Epsilon
    # plt.subplot(2, 1, 2)
    # plt.scatter(episodes, epsilons, s=10, c='green', alpha=0.6, label='Epsilon')
    # plt.title('Epsilon Decay per Episode (Scatter)')
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    # plt.grid(True)

    plt.tight_layout()
    plt.savefig(figure_name, dpi=300)
    plt.show()

# Contoh pemanggilan
if __name__ == '__main__':
    figure_name = input("Masukkan nama file untuk menyimpan plot (default: episode_data_plot.png): ") + ".png"
    if not figure_name:
        figure_name = 'episode_data_plot.png'
    plot_episode_data("bellman_episode_data_gamma_01.json", figure_name=figure_name)
