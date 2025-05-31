import numpy as np
import pandas as pd
import gym
from random import randint

from gym import spaces
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TrainTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1500, lookback_window_size=50, test_mode=False):
        super(TrainTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.df_total_steps = len(self.df) - 10
        self.current_price = None
        self.bet = 20
        self.taker_commission = 0.0055
        self.maker_commission = 0.002

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.001, 0.0041, 0.1, 0.3], dtype=np.float32), # type[long/short/nothing], entry_price, exit_price, leverage, finish
            high=np.array([2.0, 0.01, 0.5, 1.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )
        self.state_size = (lookback_window_size, 9)
        self.observation_space = spaces.Box( low=-np.inf, high=np.inf, shape=self.state_size, dtype=np.float32)
        self.order_list = []
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.balance = None
        self.prev_balance = None
        self.start_step = None
        self.end_step = None
        self.current_step = None
        self.max_episode_length = 1000
        self.test_mode = test_mode

    def reset(self, *, seed=None, options=None):

        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        if not self.test_mode:
            # рандомный старт
            start = randint(self.lookback_window_size, len(self.df) - self.max_episode_length)
            self.current_step = start
            self.start_step = start
            self.end_step = start + self.max_episode_length
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            self.current_step = self.start_step

        self.order_list.clear()
        self.market_history.clear()

        for i in range(self.lookback_window_size):
            step = self.current_step - self.lookback_window_size + i
            self.market_history.append(self._get_market_row(step))

        obs = self._next_observation()
        return obs, {}

    def _get_market_row(self, step):
        row = self.df.loc[step]
        return [row['predicted_price_1'], row['predicted_price_2'], row['predicted_price_5'], row['predicted_price_10'], row['predicted_log_10'], row['close'],
                row['high'], row['low'], row['volume']]

    def _next_observation(self):
        self.market_history.append(self._get_market_row(self.current_step))
        market_obs = np.array(self.market_history, dtype=np.float32)
        return market_obs

    def handle_action(self, act, entry_price, exit_price, leverage, stop_loss):
        second = self.df.loc[self.current_step + 1]
        third = self.df.loc[self.current_step + 2]

        notional = 30 * leverage
        reward = 0.0

        if act == 0:  # Long
            future_min = min(second['low'], third['low'])
            if future_min <= entry_price:
                reward += 0.0005  # вошёл
                for i in range(1, stop_loss): # stop_loss от 3 до 10
                    candle_i = self.df.loc[self.current_step + (i+1)] #начинаем проверять с 3 свечи
                    future_candle = candle_i['high']
                    if future_candle >= exit_price:
                        pnl = (exit_price - entry_price) * notional
                        fee_in = notional * self.maker_commission
                        fee_out = notional * self.maker_commission
                        nag = (pnl - fee_in - fee_out) + 0.001
                        reward += nag
                        #print(f"limit long reward:{nag}")
                        self.current_step += i + 1
                        break

                else: # тогда выходим по маркет через n свечей (stop_loss)
                    candle_stop = self.df.loc[self.current_step + stop_loss]
                    taker_exit_price = round(candle_stop['close'], 4)
                    pnl = (taker_exit_price - entry_price) * notional
                    fee_in = notional * self.maker_commission
                    fee_out = notional * self.taker_commission
                    nag = (pnl - fee_in - fee_out) -0.001
                    reward += nag
                    #print(f"market long reward:{nag}")
                    self.current_step += stop_loss
            else:
                reward += -0.0005 # не вошёл


        elif act == 1:  # Short
            future_max = max(second['high'], third['high'])
            if future_max >= entry_price:
                reward += 0.001  # вошёл
                for i in range(1, stop_loss):
                    candle_i = self.df.loc[self.current_step + (i + 1)]  # начинаем проверять с 3 свечи
                    future_candle = candle_i['low']
                    if future_candle <= exit_price:
                        pnl = (entry_price - exit_price) * notional
                        fee_in = notional * self.maker_commission
                        fee_out = notional * self.maker_commission
                        nag = (pnl - fee_in - fee_out) + 0.01
                        reward += nag
                        #print(f"limit short reward:{nag}")
                        self.current_step += i + 1
                        break
                else:
                    candle_stop = self.df.loc[self.current_step + stop_loss]
                    taker_exit_price = round(candle_stop['close'], 4)
                    pnl = (entry_price - taker_exit_price) * notional
                    fee_in = notional * self.maker_commission
                    fee_out = notional * self.taker_commission
                    nag = (pnl - fee_in - fee_out) - 0.01
                    reward += nag
                    #print(f"market short reward:{nag}")
                    self.current_step += stop_loss
            else:
                reward += -0.001  # не вошёл
        self.balance += reward


    def step(self, action):
        self.current_step += 1

        # type[long/short/nothing], entry_price, exit_price, leverage, stop_loss
        act_type = int(np.clip(np.round(action[0]), 0, 2))
        if act_type != 2:
            cur_candel = self.df.loc[self.current_step]
            current_price = round(cur_candel['close'], 4)
            entry_offset = np.clip(action[1], 0.001, 0.01)
            exit_offset = np.clip(action[2], 0.0041, 0.3)

            if act_type == 0:  # Лонг
                entry_price = round(current_price * (1 - entry_offset), 4)
                exit_price = round(entry_price * (1 + exit_offset), 4)
            else:  # Шорт
                entry_price = round(current_price * (1 + entry_offset), 4)
                exit_price = round(entry_price * (1 - exit_offset), 4)

            raw_leverage = np.clip(action[3], 0.1, 1)
            leverage = int((raw_leverage - 0.1) / 0.9 * 99 + 1)
            stop_loss = int(action[4] * 10)

            self.handle_action(act_type, entry_price, exit_price, leverage, stop_loss)

        _reward = self.balance - self.prev_balance
        self.prev_balance = self.balance

        # ==== УСЛОВИЕ КОНЦА ====
        _terminated = self.current_step >= self.end_step or self.balance <= self.initial_balance / 2
        # ==== СОСТОЯНИЕ ====
        _obs = self._next_observation()
        return _obs, _reward, _terminated, False, {}


if __name__ == '__main__':

    df = pd.read_pickle('test_rl_df.pkl')
    #df = df.drop(['turnover', 'timestamp'], axis=1)
    df_train = df.iloc[:40000].reset_index(drop=True)
    df_test = df.iloc[40000:].reset_index(drop=True)
    #----------------
    n_envs = 4
    # Обёртка для создания одной среды
    def make_env(df, i):
        def _init():
            env = TrainTradingEnv(df=df.copy())
            env.name = f"env_{i}"
            return env
        return _init
    # Векторизованные среды для обучения
    envs = DummyVecEnv([make_env(df_train, i) for i in range(n_envs)])
    # Создание и обучение модели
    model = PPO("MlpPolicy", envs, verbose=1) # MlpPolicy / MultiInputPolicy
    model.learn(total_timesteps=1000)
    #--------------------
    model.save("test_pro_ethmodel")
    print("✅ Обучение завершено и модель сохранена.")

    env_test = TrainTradingEnv(df_test, test_mode=True)
    obs, _ = env_test.reset()

    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated

    # Финальные результаты
    print(f"📊 Финальный баланс на тест данных: {env_test.balance:.2f}")
    print(f"💸 Общая награда на тест данных: {total_reward:.2f}")