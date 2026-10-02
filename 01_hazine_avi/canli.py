"""
Hazine Avı: canlı izleme

Robotu bir pencerede adım adım izle:
    1) Eğitimden önce: hiçbir şey bilmiyor, rastgele dolaşıp tuzağa düşüyor.
    2) Eğitim: 500 bölüm oynarken beyninin (Q-tablosunun) nasıl dolduğunu görüyorsun.
    3) Eğitimden sonra: öğrendiği yoldan hazineye gidiyor.

Çalıştırmak için:
    conda activate pytorch_env
    python canli.py

Ortam, ajan ve eğitim hazine_avi.py'den geliyor; bu dosya yalnızca izlemeyi sağlıyor.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch

import hazine_avi as h
from gorsel import ARROWS, INK, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE

# Hız ayarları: animasyonu hızlı ya da yavaş bulursan bunlarla oyna
FRAMES_PER_STEP = 10      # robot bir kareden diğerine kaç ara görüntüyle kaysın
FRAME_DELAY = 0.03        # ara görüntüler arasındaki bekleme (saniye)
PAUSE = 1.5               # bir bölüm bitince ya da yeni bir perde başlayınca bekleme (saniye)
TRAIN_DRAW_EVERY = 5      # eğitimde ekran kaç bölümde bir güncellensin
TRAIN_FRAME_DELAY = 0.05  # eğitim görüntüleri arasındaki bekleme (saniye)
LIVE_MAX_STEPS = 30       # izlerken bir bölüm en fazla kaç adım sürsün (takılan robotu beklememek için)

TRAP_COLOR = "#d03b3b"
TREASURE_COLOR = "#eda100"
# Karenin değeri arttıkça açık griden koyu maviye
VALUE_COLORS = LinearSegmentedColormap.from_list(
    "deger", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
VALUE_SCALE = Normalize(vmin=0, vmax=h.REWARD_TREASURE, clip=True)


# =============================================================================
# Pencere: haritayı, robotu ve robotun beynini çizer
# =============================================================================

class LiveBoard:
    """Haritayı, robotu ve robotun beynini (Q-tablosunu) çizen pencere."""

    def __init__(self, env):
        self.env = env
        self.fig = plt.figure(figsize=(6.4, 7.6), facecolor=SURFACE)
        self.fig.canvas.manager.set_window_title("Hazine Avı: canlı")
        self.title = self.fig.text(0.06, 0.955, "", fontsize=15, fontweight="bold", color=INK)
        self.subtitle = self.fig.text(0.06, 0.93, "", fontsize=10, color=INK_SECONDARY,
                                      va="top", linespacing=1.4)
        self.status = self.fig.text(0.5, 0.145, "", ha="center", fontsize=11, color=INK)

        self.ax = self.fig.add_axes([0.06, 0.18, 0.88, 0.68])
        self.ax.set_xlim(-0.5, env.n_cols - 0.5)
        self.ax.set_ylim(env.n_rows - 0.5, -0.5)   # satır 0 en üstte olsun
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Kareler: tuzak ve hazine sabit renkte; boş kareler robotun beynine göre boyanacak
        self.tiles, self.arrows, self.values = {}, {}, {}
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                tile = FancyBboxPatch((col - 0.46, row - 0.46), 0.92, 0.92,
                                      boxstyle="round,pad=0,rounding_size=0.08", linewidth=0)
                self.ax.add_patch(tile)
                cell = env.grid[row][col]
                if cell == "X":
                    tile.set_facecolor(TRAP_COLOR)
                    self.ax.text(col, row, "X", ha="center", va="center",
                                 fontsize=24, fontweight="bold", color="white")
                elif cell == "$":
                    tile.set_facecolor(TREASURE_COLOR)
                    self.ax.text(col, row, "$", ha="center", va="center",
                                 fontsize=26, fontweight="bold", color=INK)
                else:
                    self.tiles[(row, col)] = tile
                    self.arrows[(row, col)] = self.ax.text(col, row - 0.04, "", ha="center",
                                                           va="center", fontsize=22)
                    self.values[(row, col)] = self.ax.text(col, row + 0.33, "", ha="center",
                                                           va="center", fontsize=9)

        # Robot: turuncu bir top, iki göz ve arkasında bıraktığı iz
        (self.trail,) = self.ax.plot([], [], color=ORANGE, linewidth=4, alpha=0.45,
                                     solid_capstyle="round", zorder=2)   # okların altında kalsın
        self.body = Circle((0, 0), 0.27, facecolor=ORANGE, edgecolor=SURFACE,
                           linewidth=2, zorder=5)
        self.eyes = [Circle((0, 0), 0.045, facecolor="white", zorder=6) for _ in range(2)]
        for part in [self.body, *self.eyes]:
            self.ax.add_patch(part)

        # Bölüm sonunda ortada çıkan mesaj kutusu
        self.message = self.ax.text((env.n_cols - 1) / 2, (env.n_rows - 1) / 2, "",
                                    ha="center", va="center", fontsize=17, fontweight="bold",
                                    color=INK, zorder=10, visible=False, linespacing=1.5,
                                    bbox=dict(boxstyle="round,pad=0.6", facecolor=SURFACE,
                                              edgecolor=INK_MUTED, alpha=0.95))

        # Renk ölçeği: hangi rengin hangi değere karşılık geldiği
        cax = self.fig.add_axes([0.2, 0.07, 0.6, 0.022])
        bar = self.fig.colorbar(ScalarMappable(VALUE_SCALE, VALUE_COLORS), cax=cax,
                                orientation="horizontal")
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
        bar.set_label("Karenin değeri: robot o kareden kaç puan toplamayı umuyor",
                      color=INK_SECONDARY, fontsize=9)

        plt.show(block=False)

    def set_header(self, title, subtitle):
        self.title.set_text(title)
        self.subtitle.set_text(subtitle)

    def set_status(self, text):
        self.status.set_text(text)

    def show_brain(self, agent, show_arrows=True):
        """Boş kareleri Q-tablosuna göre boyar: koyu mavi = robotun gözünde değerli kare."""
        for position, tile in self.tiles.items():
            state = self.env.to_state(position)
            q_values = agent.q_table[state]
            color = VALUE_COLORS(VALUE_SCALE(q_values.max()))
            tile.set_facecolor(color)
            red, green, blue, _ = color
            is_dark = 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
            text_color = "white" if is_dark else INK
            self.values[position].set(text=f"{q_values.max():.1f}", color=text_color)
            # Hamlelerin değerleri hep aynıysa robot bu kare hakkında henüz bir şey öğrenmemiştir
            learned = q_values.max() > q_values.min()
            arrow = ARROWS[agent.best_action(state)] if show_arrows and learned else ""
            self.arrows[position].set(text=arrow, color=text_color)

    def place_robot(self, position, facing=(0, 0)):
        """Robotu (satır, sütun) konumuna koyar; gözleri gittiği yöne bakar."""
        row, col = position
        d_row, d_col = facing
        self.body.center = (col, row)
        self.eyes[0].center = (col - 0.09 + 0.05 * d_col, row - 0.05 + 0.05 * d_row)
        self.eyes[1].center = (col + 0.09 + 0.05 * d_col, row - 0.05 + 0.05 * d_row)

    def hide_robot(self):
        """Robotu, izini ve mesaj kutusunu gizler (eğitim sırasında)."""
        for part in [self.body, *self.eyes, self.trail, self.message]:
            part.set_visible(False)

    def start_episode(self, position):
        """Yeni bölüm: robotu başlangıç konumuna koyar, eski izi ve mesajı siler."""
        self.message.set_visible(False)
        self.body.set(radius=0.27, facecolor=ORANGE)
        for part in [self.body, *self.eyes, self.trail]:
            part.set_visible(True)
        self.place_robot(position)
        self.trail.set_data([position[1]], [position[0]])

    def move_robot(self, start, end, action):
        """Robotu bir kareden diğerine kaydırarak götürür."""
        facing = self.env.ACTIONS[action]
        if start == end:
            # Duvara çarptı: duvara doğru biraz gidip geri sekiyor
            target = (start[0] + 0.3 * facing[0], start[1] + 0.3 * facing[1])
            half = np.linspace(0, 1, FRAMES_PER_STEP // 2 + 1)
            fractions = np.concatenate([half, half[::-1]])
        else:
            target = end
            fractions = np.linspace(0, 1, FRAMES_PER_STEP + 1)
        for t in fractions[1:]:
            self.place_robot((start[0] + (target[0] - start[0]) * t,
                              start[1] + (target[1] - start[1]) * t), facing)
            self.wait(FRAME_DELAY)
        if start != end:
            cols, rows = self.trail.get_data()
            self.trail.set_data([*cols, end[1]], [*rows, end[0]])

    def end_episode(self, text, fell_into_trap):
        """Bölümün sonucunu ortada bir mesajla gösterir."""
        if fell_into_trap:
            self.body.set(radius=0.2, facecolor=INK_MUTED)   # çukura düştü
        self.message.set(text=text, visible=True)
        self.wait(PAUSE)

    def wait(self, seconds):
        """Ekranı günceller ve biraz bekler. Pencere kapatıldıysa programı bitirir."""
        if not plt.fignum_exists(self.fig.number):
            sys.exit()
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(max(seconds, 0.001))


# =============================================================================
# Canlı oynatma ve eğitim
# =============================================================================

def play_live(board, env, choose_action, label, random_start=False):
    """Bir bölümü pencerede canlı oynatır (öğrenmeden)."""
    state = env.reset(random_start=random_start)
    board.start_episode(env.position)
    board.set_status(f"{label}   ·   adım 0   ·   toplam ödül: 0")
    board.wait(PAUSE / 2)

    total_reward = 0
    for step in range(1, LIVE_MAX_STEPS + 1):
        action = choose_action(state)            # 1) robot bir hamle seçer
        old_position = env.position
        state, reward, done = env.step(action)   # 2) ortam robotu hareket ettirir, ödülü verir
        total_reward += reward
        board.move_robot(old_position, env.position, action)
        board.set_status(f"{label}   ·   adım {step}   ·   toplam ödül: {total_reward:+g}")
        if done:
            break

    row, col = env.position
    if env.grid[row][col] == "$":
        board.end_episode(f"HAZİNEYİ BULDU!\n{step} adım · toplam ödül {total_reward:+g}", False)
    elif env.grid[row][col] == "X":
        board.end_episode(f"TUZAĞA DÜŞTÜ!\n{step}. adımda · toplam ödül {total_reward:+g}", True)
    else:
        board.end_episode(f"SÜRE DOLDU\n{step} adımda bir yere varamadı", False)


def train_live(board, env, agent):
    """Robotu eğitir; birkaç bölümde bir robotun beynini (Q-tablosunu) ekrana çizer."""
    board.hide_robot()
    successes = []
    for episode in range(1, h.N_EPISODES + 1):
        epsilon = agent.epsilon
        total_reward, found_treasure = h.train_episode(env, agent)   # bir bölüm oyna ve öğren
        successes.append(found_treasure)
        if episode % TRAIN_DRAW_EVERY == 0 or episode == h.N_EPISODES:
            board.show_brain(agent)
            board.set_status(f"Bölüm {episode}/{h.N_EPISODES}   ·   ε = {epsilon:.2f}   ·   "
                             f"son 50 bölümde başarı: %{100 * np.mean(successes[-50:]):.0f}")
            board.wait(TRAIN_FRAME_DELAY)


def main():
    env, agent = h.create_env_and_agent()   # hazine_avi.py ile aynı kurulum, aynı sonuçlar
    board = LiveBoard(env)

    board.set_header("1) Eğitimden önce",
                     "Robot hiçbir şey bilmiyor: her hamlesi rastgele (ε = 1).\n"
                     "Beyni bomboş, bütün karelerin değeri 0.")
    board.show_brain(agent, show_arrows=False)
    for attempt in range(1, 4):
        play_live(board, env, agent.choose_action, f"Deneme {attempt}/3")

    board.set_header("2) Eğitim",
                     f"Robot {h.N_EPISODES} bölüm oynuyor; çok hızlı olduğu için hareketleri gizli.\n"
                     "İzle: hazinenin değeri adım adım geriye yayılıyor, oklar beliriyor.")
    train_live(board, env, agent)
    board.wait(PAUSE)

    board.set_header("3) Eğitimden sonra",
                     "Artık rastgele hamle yok (ε = 0): robot her karede okun gösterdiği\n"
                     "yöne, yani en değerli komşuya gidiyor.")
    play_live(board, env, agent.best_action, "Başlangıç karesinden")
    for attempt in range(2):
        play_live(board, env, agent.best_action, "Rastgele bir kareden", random_start=True)

    board.set_status("Bitti! Pencereyi kapatınca program sona erer.")
    plt.show()   # pencere kapanana kadar açık kalsın


if __name__ == "__main__":
    main()
