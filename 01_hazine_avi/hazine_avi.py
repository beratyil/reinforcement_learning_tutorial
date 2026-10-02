"""
Hazine Avı: Reinforcement Learning "Hello World"

Bir robot 5x5'lik bir ızgarada hazineyi arıyor. Haritayı BİLMİYOR: nerede
tuzak, nerede hazine var, hiçbir fikri yok. Tek bildiği, her hamlesinden sonra
ortamın ona verdiği puan (ödül). Q-Learning algoritmasıyla, deneme-yanılma
yoluyla hazineye giden en kısa ve güvenli yolu kendi kendine öğreniyor.

Çalıştırmak için:
    conda activate pytorch_env
    python hazine_avi.py
"""

from pathlib import Path

import numpy as np

import gorsel

# =============================================================================
# AYARLAR: Bunlarla oynamaktan çekinme! Değiştir, tekrar çalıştır, sonucu gör.
# =============================================================================

# Harita:  R = robot (başlangıç)   X = tuzak   $ = hazine   . = boş kare
GRID = [
    "R.X..",
    "X.X..",
    "....X",
    ".XX.X",
    "....$",
]

# Ödüller: ortamın robota verdiği puanlar. Robot dünyayı SADECE bunlar sayesinde tanır.
REWARD_TREASURE = 10   # hazineyi buldu  → bölüm biter
REWARD_TRAP = -10      # tuzağa düştü    → bölüm biter
REWARD_STEP = -1       # her normal adım → "oyalanma, acele et!"

# Q-Learning ayarları (hiperparametreler)
ALPHA = 0.2            # öğrenme hızı: her yeni deneyim eski bilgiyi ne kadar değiştirsin? (0-1)
GAMMA = 0.95           # indirim oranı: gelecekteki ödüller ne kadar önemli? (0-1)
EPSILON_START = 1.0    # keşif oranı: başta hamlelerin %100'ü rastgele
EPSILON_MIN = 0.01     # en az %1 rastgelelik her zaman kalsın
EPSILON_DECAY = 0.99   # her bölüm sonunda epsilon bu sayıyla çarpılır (yavaş yavaş azalır)

N_EPISODES = 500       # kaç bölüm (oyun) oynayarak öğrenecek
MAX_STEPS = 100        # bir bölümde en fazla kaç adım atılabilir
RANDOM_START = True    # eğitimde her bölüm rastgele bir boş kareden başlasın mı?
                       # (robot haritanın her köşesini tanısın diye; False yapıp farkı gör!)
SEED = 42              # rastgeleliği sabitler: her çalıştırmada aynı sonuç çıkar


# =============================================================================
# 1) ORTAM (Environment): robotun içinde yaşadığı dünya
# =============================================================================

class TreasureHuntEnv:
    """Izgara şeklinde bir dünya.

    Gymnasium kütüphanesindeki ortamlarla aynı mantıkta iki metodu var:
        reset()      → yeni bölüm başlatır, başlangıç durumunu döndürür
        step(action) → robotu hareket ettirir, (yeni_durum, ödül, bitti_mi) döndürür
    (Gymnasium bunlara ek olarak birkaç değer daha döndürür; burada sade tuttuk.)
    """

    # Robotun yapabileceği 4 eylem. Her biri: (satır değişimi, sütun değişimi)
    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 0: ↑ yukarı  1: → sağ  2: ↓ aşağı  3: ← sol

    def __init__(self, grid, rng):
        self.grid = grid
        self.n_rows = len(grid)
        self.n_cols = len(grid[0])
        self.n_states = self.n_rows * self.n_cols   # her kare bir durum: 5x5 = 25 durum
        self.n_actions = len(self.ACTIONS)          # 4 eylem
        self.rng = rng                              # rastgele sayı üreteci

        # Haritayı tara: başlangıç karesini (R) ve robotun durabileceği boş kareleri bul
        self.free_cells = []
        for row, line in enumerate(grid):
            for col, cell in enumerate(line):
                if cell == "R":
                    self.start = (row, col)
                if cell in "R.":
                    self.free_cells.append((row, col))
        self.position = self.start

    def to_state(self, position):
        """(satır, sütun) konumunu tek bir sayıya çevirir: 0, 1, 2, ..., 24.
        Bu sayı, Q-tablosunda o karenin satır numarası olacak."""
        row, col = position
        return row * self.n_cols + col

    def reset(self, random_start=False):
        """Yeni bir bölüm başlatır: robotu başlangıç karesine
        (random_start=True ise rastgele bir boş kareye) koyar."""
        if random_start:
            self.position = self.free_cells[self.rng.integers(len(self.free_cells))]
        else:
            self.position = self.start
        return self.to_state(self.position)

    def step(self, action):
        """Robotu bir adım hareket ettirir ve sonucu bildirir."""
        d_row, d_col = self.ACTIONS[action]
        # Izgaranın dışına çıkamaz: kenara çarparsa olduğu yerde kalır
        row = min(max(self.position[0] + d_row, 0), self.n_rows - 1)
        col = min(max(self.position[1] + d_col, 0), self.n_cols - 1)
        self.position = (row, col)

        cell = self.grid[row][col]
        if cell == "$":
            reward, done = REWARD_TREASURE, True
        elif cell == "X":
            reward, done = REWARD_TRAP, True
        else:
            reward, done = REWARD_STEP, False
        return self.to_state(self.position), reward, done


# =============================================================================
# 2) AJAN (Agent): öğrenen robot
# =============================================================================

class QLearningAgent:
    """Q-Learning ile öğrenen ajan.

    Robotun bütün "beyni" tek bir tablodur, adı Q-tablosu:
        satırlar = durumlar (25 kare),  sütunlar = eylemler (↑ → ↓ ←)
        q_table[durum, eylem] = "Bu karede bu hamleyi yaparsam, bundan sonra
                                 toplam kaç puan toplarım?" sorusuna verdiği tahmin.
    """

    def __init__(self, n_states, n_actions, alpha, gamma,
                 epsilon, epsilon_min, epsilon_decay, rng):
        self.q_table = np.zeros((n_states, n_actions))   # başta hiçbir şey bilmiyor: hepsi 0
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng   # rastgele sayı üreteci

    def choose_action(self, state):
        """Epsilon-greedy seçim: yeni şeyler mi denesin, bildiğini mi kullansın?"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))   # KEŞFET: rastgele bir hamle dene
        return self.best_action(state)                      # KULLAN: en iyi bildiğin hamleyi yap

    def best_action(self, state):
        """Bu durumda Q-değeri en yüksek olan eylem."""
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning güncellemesi, algoritmanın kalbi:

            Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',·) − Q(s,a) ]
                                    └──────── hedef ────────┘
        """
        # Vardığım karede yapabileceğim en iyi hamlenin değeri (bölüm bittiyse gelecek yok: 0)
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        # Hedef: şimdi aldığım ödül + gelecekte toplamayı umduğum ödül (γ ile indirimli)
        target = reward + self.gamma * best_next
        # Hata: hedef ile şu anki tahminim arasındaki fark
        error = target - self.q_table[state, action]
        # Tahminimi hedefe doğru α kadar kaydır
        self.q_table[state, action] += self.alpha * error

    def decay_epsilon(self):
        """Keşif oranını biraz azalt: robot öğrendikçe daha az rastgele davranır."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# 3) EĞİTİM: ajan ile ortam arasındaki döngü
# =============================================================================

def train_episode(env, agent):
    """Tek bir eğitim bölümü oynatır: robot attığı her adımdan öğrenir.
    Bölümde toplanan ödülü ve hazinenin bulunup bulunmadığını döndürür."""
    state = env.reset(random_start=RANDOM_START)
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.choose_action(state)                    # 1) ajan bir eylem seçer
        next_state, reward, done = env.step(action)            # 2) ortam yeni durumu ve ödülü verir
        agent.learn(state, action, reward, next_state, done)   # 3) ajan bu deneyimden öğrenir
        state = next_state
        total_reward += reward
        if done:
            break

    agent.decay_epsilon()   # bölüm bitti: bir sonrakinde biraz daha az rastgele davransın
    found_treasure = (reward == REWARD_TREASURE)   # son ödül hazine ödülüyse hazineyi bulmuştur
    return total_reward, found_treasure


def train(env, agent):
    """Robotu N_EPISODES bölüm boyunca eğitir ve her bölümün sonucunu kaydeder."""
    rewards = []     # her bölümde toplanan toplam ödül
    successes = []   # her bölümde hazine bulundu mu? (True / False)
    epsilons = []    # her bölümde kullanılan keşif oranı

    for episode in range(N_EPISODES):
        epsilons.append(agent.epsilon)
        total_reward, found_treasure = train_episode(env, agent)
        rewards.append(total_reward)
        successes.append(found_treasure)

    return rewards, successes, epsilons


def play_episode(env, choose_action):
    """Başlangıç karesinden bir bölüm oynatır ama ÖĞRENMEZ.
    Robotun geçtiği kareleri ve hamlelerini döndürür."""
    state = env.reset()
    path = [env.position]
    actions = []
    total_reward = 0
    for step in range(MAX_STEPS):
        action = choose_action(state)
        state, reward, done = env.step(action)
        path.append(env.position)
        actions.append(action)
        total_reward += reward
        if done:
            break
    return path, actions, total_reward


# =============================================================================
# 4) ANA PROGRAM
# =============================================================================

def create_env_and_agent():
    """Yukarıdaki ayarlarla ortamı ve ajanı oluşturur."""
    rng = np.random.default_rng(SEED)
    env = TreasureHuntEnv(GRID, rng)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        rng=rng,
    )
    return env, agent


def main():
    env, agent = create_env_and_agent()

    gorsel.print_title("HARİTA")
    gorsel.print_board(env, lambda row, col: env.grid[row][col])
    print(f"\n   R = robot   X = tuzak ({REWARD_TRAP})   $ = hazine (+{REWARD_TREASURE})"
          f"   her adım: {REWARD_STEP}")

    gorsel.print_title("1) EĞİTİMDEN ÖNCE: robot hiçbir şey bilmiyor, rastgele dolaşıyor")
    # epsilon şu an 1 olduğu için choose_action tamamen rastgele hamle seçer
    for attempt in range(1, 4):
        path, actions, total_reward = play_episode(env, agent.choose_action)
        gorsel.print_episode(env, path, actions, total_reward, label=f"{attempt}. deneme")

    gorsel.print_title(f"2) EĞİTİM: {N_EPISODES} bölüm oynuyor, her adımdan öğreniyor")
    rewards, successes, epsilons = train(env, agent)
    gorsel.print_progress(rewards, successes, epsilons)

    gorsel.print_title("3) EĞİTİMDEN SONRA: her karede en iyi bildiği hamleyi yapıyor")
    path, actions, total_reward = play_episode(env, agent.best_action)
    gorsel.print_episode(env, path, actions, total_reward)
    gorsel.print_path(env, path, actions)

    gorsel.print_title("4) ROBOTUN BEYNİ: Q-tablosu")
    gorsel.print_start_q_values(env, agent)
    gorsel.print_value_map(env, agent)
    gorsel.print_policy_map(env, agent)

    plot_file = Path(__file__).with_name("ogrenme_egrisi.png")
    gorsel.plot_learning_curve(rewards, successes, epsilons, plot_file)
    print(f"\n   Öğrenme eğrisi kaydedildi: {plot_file.name}\n")


if __name__ == "__main__":
    main()
