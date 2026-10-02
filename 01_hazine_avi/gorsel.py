"""
Ekrana yazdırma ve grafik çizme yardımcıları.

Bu dosyada Reinforcement Learning yok; sadece sonuçları göstermeye yarıyor.
RL'i anlamak için hazine_avi.py yeterli, bu dosyayı okumak zorunda değilsin.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ARROWS = ["↑", "→", "↓", "←"]                     # eylem numarası → ok işareti
ACTION_NAMES = ["yukarı", "sağ", "aşağı", "sol"]

# Grafik renkleri
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"


# -----------------------------------------------------------------------------
# Terminale yazdırma
# -----------------------------------------------------------------------------

def print_title(text):
    print()
    print("=" * 66)
    print(f"  {text}")
    print("=" * 66)


def print_board(env, cell_text, width=3):
    """Izgarayı satır satır yazdırır. cell_text(satır, sütun) her karede ne yazacağını söyler."""
    for row in range(env.n_rows):
        print("   " + "".join(cell_text(row, col).center(width) for col in range(env.n_cols)))


def print_episode(env, path, actions, total_reward, label="Hamleler"):
    """Bir bölümde yapılan hamleleri ve bölümün nasıl bittiğini yazdırır."""
    row, col = path[-1]
    last_cell = env.grid[row][col]
    if last_cell == "$":
        outcome = f"{len(actions)}. adımda HAZİNEYİ BULDU!"
    elif last_cell == "X":
        outcome = f"{len(actions)}. adımda TUZAĞA DÜŞTÜ!"
    else:
        outcome = f"{len(actions)} adım attı ama bir yere varamadı."
    moves = " ".join(ARROWS[a] for a in actions[:30])   # çok uzunsa ilk 30 hamleyi göster
    if len(actions) > 30:
        moves += f" ... (+{len(actions) - 30} hamle daha)"
    prefix = f"   {label}: "
    print(prefix + moves)
    print(" " * len(prefix) + f"sonuç: {outcome}  (toplam ödül: {total_reward:+g})")


def print_path(env, path, actions):
    """Robotun izlediği yolu harita üzerinde oklarla gösterir."""
    arrow_at = {position: ARROWS[action] for position, action in zip(path, actions)}
    print("\n   İzlediği yol:")
    print_board(env, lambda row, col: arrow_at.get((row, col), env.grid[row][col]))


def print_progress(rewards, successes, epsilons, every=50):
    """Eğitimin ilerleyişini her 50 bölümde bir, tablo satırı olarak yazdırır."""
    every = min(every, len(rewards))   # 50'den az bölüm oynandıysa
    print(f"   {'Bölüm':^7}{'Hazineyi bulma':^18}{'Ortalama ödül':^18}{'Epsilon':^15}")
    print(f"   {'':^7}{f'(son {every} bölüm)':^18}{f'(son {every} bölüm)':^18}{'(keşif oranı)':^15}")
    for end in range(every, len(rewards) + 1, every):
        success_rate = 100 * np.mean(successes[end - every:end])
        mean_reward = np.mean(rewards[end - every:end])
        print(f"   {end:^7}{f'%{success_rate:.0f}':^18}{f'{mean_reward:+.1f}':^18}"
              f"{f'{epsilons[end - 1]:.2f}':^15}")


def print_start_q_values(env, agent):
    """Başlangıç karesinin Q-tablosundaki satırını gösterir."""
    state = env.to_state(env.start)
    best = agent.best_action(state)
    print(f"   Başlangıç karesi (durum {state}) için Q-değerleri:")
    for action, q_value in enumerate(agent.q_table[state]):
        note = "   ← en büyük: robot bunu seçer" if action == best else ""
        print(f"      {ARROWS[action]} {ACTION_NAMES[action]:<7}{q_value:+7.2f}{note}")


def print_value_map(env, agent):
    """Her karedeki en büyük Q-değeri: robotun gözünde o karenin ne kadar 'iyi' olduğu."""
    def cell_text(row, col):
        if env.grid[row][col] in "X$":
            return env.grid[row][col]
        return f"{agent.q_table[env.to_state((row, col))].max():.1f}"

    print("\n   Değer haritası (her karedeki en büyük Q-değeri):")
    print_board(env, cell_text, width=7)


def print_policy_map(env, agent):
    """Her karede robotun seçeceği hamle. Öğrenilen POLİTİKA tam olarak budur."""
    def cell_text(row, col):
        if env.grid[row][col] in "X$":
            return env.grid[row][col]
        return ARROWS[agent.best_action(env.to_state((row, col)))]

    print("\n   Politika haritası (her karede seçeceği hamle):")
    print_board(env, cell_text)


# -----------------------------------------------------------------------------
# Grafik
# -----------------------------------------------------------------------------

def moving_average(values, window):
    """Kayan ortalama: her noktada son `window` değerin ortalaması."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def style_axes(ax, title, subtitle, legend_loc):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0, pad=6)
    ax.text(0, 1.13, title, transform=ax.transAxes, color=INK, fontsize=12, fontweight="bold")
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)


def label_end(ax, x, y, text, color):
    """Çizginin sonuna bir nokta ve değer etiketi koyar."""
    ax.plot(x, y, "o", markersize=7, color=color,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points",
                va="center", color=INK, fontsize=9, fontweight="bold")


def plot_learning_curve(rewards, successes, epsilons, filename, window=50):
    """Eğitim boyunca ödülün ve başarı oranının nasıl değiştiğini çizer, PNG olarak kaydeder."""
    window = min(window, len(rewards))   # 50'den az bölüm oynandıysa
    episodes = np.arange(1, len(rewards) + 1)
    avg_episodes = episodes[window - 1:]
    avg_reward = moving_average(rewards, window)
    success_rate = 100 * moving_average(np.array(successes, dtype=float), window)
    epsilon_rate = 100 * np.array(epsilons)

    fig, (ax_reward, ax_rate) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                             facecolor=SURFACE)

    # Üst: her bölümde toplanan toplam ödül
    ax_reward.plot(episodes, rewards, color=BASELINE, linewidth=1, label="her bölüm")
    ax_reward.plot(avg_episodes, avg_reward, color=BLUE, linewidth=2,
                   label=f"son {window} bölümün ortalaması")
    label_end(ax_reward, avg_episodes[-1], avg_reward[-1], f"{avg_reward[-1]:+.1f}", BLUE)
    style_axes(ax_reward, "Bölüm başına toplam ödül",
               "Robot öğrendikçe daha az tuzağa düşüyor ve hazineye daha kısa yoldan gidiyor",
               legend_loc="lower right")

    # Alt: hazineyi bulma oranı ve rastgele hamle oranı (ikisi de yüzde → aynı eksen)
    ax_rate.plot(avg_episodes, success_rate, color=BLUE, linewidth=2,
                 label=f"hazineyi bulma oranı (son {window} bölüm)")
    ax_rate.plot(episodes, epsilon_rate, color=ORANGE, linewidth=2,
                 label="rastgele hamle oranı (epsilon)")
    label_end(ax_rate, avg_episodes[-1], success_rate[-1], f"%{success_rate[-1]:.0f}", BLUE)
    label_end(ax_rate, episodes[-1], epsilon_rate[-1], f"%{epsilon_rate[-1]:.0f}", ORANGE)
    ax_rate.set_ylim(0, 105)
    ax_rate.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"%{value:.0f}"))
    style_axes(ax_rate, "Keşiften kullanıma",
               "Rastgele hamleler azaldıkça robot öğrendiğini kullanıyor ve hazineyi buluyor",
               legend_loc="center right")
    ax_rate.set_xlabel("Bölüm (oynanan oyun sayısı)", color=INK_MUTED, fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.93, top=0.89, bottom=0.08, hspace=0.45)
    fig.savefig(filename, dpi=150, facecolor=SURFACE)
    plt.close(fig)
