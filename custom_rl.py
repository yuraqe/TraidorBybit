import numpy as np
import pandas as pd
import gym

from gym import spaces
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class TrainTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1500, lookback_window_size=70):
        super(TrainTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.df_total_steps = len(self.df) - 1
        self.current_price = None
        self.total_age = 0
        self.bet = 20
        self.taker_commission = 0.004
        self.maker_commission = 0.002

        self.N = 2
        self.action_space = spaces.Box(
            low=np.tile([0.0, -0.05, 1.0, 0.0], self.N),  # [type, price_offset, leverage, index_or_number]
            high=np.tile([6.0, 0.05, 100.0, 1.0], self.N),
            shape=(4 * self.N,),
            dtype=np.float32
        )
        self.state_size = (lookback_window_size, 11)
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=self.state_size, dtype=np.float32),
            "orders": spaces.Box(low=0.0, high=np.inf, shape=(2, 6), dtype=np.float32)
        })
        self.limit_order_list = []
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.balance = None
        self.prev_balance = None
        self.start_step = None
        self.end_step = None
        self.current_step = None

    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        self.total_age = 0

        self.start_step = self.lookback_window_size
        self.end_step = self.df_total_steps
        self.current_step = self.start_step

        self.limit_order_list.clear()
        self.market_history.clear()

        for i in reversed(range(self.lookback_window_size)):
            step = self.current_step - i
            self.market_history.append(self._get_market_row(step))

        obs = self._next_observation()
        return obs, {}

    def _get_market_row(self, step):
        row = self.df.loc[step]
        return [row['open'], row['high'], row['low'], row['close'], row['volume'], row['SMA14'], row['RSI14'], row['OBV14'], row['ATR14'], row['MACD'], row['MACD_SIGNAL']]


    def _next_observation(self):
        self.market_history.append(self._get_market_row(self.current_step))
        market_obs = np.array(self.market_history, dtype=np.float32)

        orders_obs = np.zeros((2, 6), dtype=np.float32)
        for i, order in enumerate(self.limit_order_list[:2]):
            orders_obs[i] = [
                order["entry_price"],
                order["size"],
                order["type"],
                order["leverage"],
                self.total_age - order['age'],
                order['activated']
            ]

        return {
            "market": market_obs,
            "orders": orders_obs
        }

    def _handle_action(self, act):
        act_type, offset, lev, index = act

        if act_type == 0:
            self._open_position(0, offset, lev)
        elif act_type == 1:
            self._open_position(1, offset, lev)
        elif act_type == 3:
            self._exit_market()
        elif act_type == 5:
            self._open_position(5, offset, lev)
        elif act_type == 6:
            self._open_position(6, offset, lev)


    def step(self, action):
        self.current_step += 1
        self.total_age += 1

        current_candle = self.df.loc[self.current_step]
        self.current_price = current_candle['close']
        if len(self.limit_order_list) > 0:
            self._check_orders(current_candle)

        # ==== ПАРСИМ ДЕЙСТВИЕ ====
        parsed_actions = []
        for i in range(self.N):
            base = i * 4
            act_type = int(np.clip(np.round(action[base]), 0, 6))
            price_offset = round(np.clip(action[base + 1], -0.05, 0.05), 2)
            leverage = int(np.clip(action[base + 2], 1, 100)) # 0.1 - 1
            index_or_number = int(np.clip(action[base + 3], 0, 1))
            parsed_actions.append([act_type, price_offset, leverage, index_or_number])

        a1, a2 = parsed_actions  # [type, offset, lev, index]
        type_pair = (a1[0], a2[0])
        t1, t2 = a1[0], a2[0]

        if t1 == 4 or t2 == 4:
            if t1 == 4 and t2 == 4:
                pass
            elif t1 == 4:
                self._handle_action(a2)
            else:
                self._handle_action(a1)

        elif t1 == t2:
            self._handle_action(a1)


        elif type_pair in [(0, 6), (1, 5), (6, 0), (5, 1)]:
            pass  # long vs short, игнорируем

        elif type_pair == (0, 5) or type_pair == (1, 5): # limit/market
            self._open_position(0, a1[1], a1[2])
            self._open_position(1, a2[1], a2[2])
        elif type_pair == (5, 0) or type_pair == (1, 5):  # market/limit
            self._open_position(1, a1[1], a1[2])
            self._open_position(0, a2[1], a2[2])


        # ==== ПОДСЧЁТ НАГРАДЫ ====
        _reward = self.balance - self.prev_balance
        self.prev_balance = self.balance

        # ==== УСЛОВИЕ КОНЦА ====
        _terminated = self.current_step >= self.end_step or self.balance <= self.initial_balance / 2

        # ==== СОСТОЯНИЕ ====
        _obs = self._next_observation()
        return _obs, _reward, _terminated, False, {}

    def _check_orders(self, candle):
        if len(self.limit_order_list) > 0 and self.limit_order_list[0]["activated"] == 0:
            order = self.limit_order_list[0]
            direction = order["type"]
            entry_price = order["entry_price"]

            # Проверка исполнения лимитки
            if (direction == 0 and candle["low"] <= entry_price) or \
                    (direction == 1 and candle["high"] >= entry_price):

                order["activated"] = 1
                order["type"] = 5 if direction == 0 else 6  # маркет-лонг или маркет-шорт
                commission = entry_price * order["size"] * self.maker_commission

                if len(self.limit_order_list) > 1 and self.limit_order_list[1]["activated"] == 1:
                    market_order = self.limit_order_list[1]

                    current_size = market_order["size"]
                    added_size = order["size"]
                    new_size = current_size + added_size
                    new_entry_price = round(
                            (market_order["entry_price"] * current_size + entry_price * added_size) / new_size, 2)
                    market_order["size"] = new_size
                    market_order["entry_price"] = new_entry_price

                    self.balance -= commission  # списываем только комиссию, маржа уже была внесена ранее
                    self.limit_order_list.pop(0)

                else:
                    # Нет активной позиции — списываем маржу + комиссию
                    margin_required = (order["size"] * entry_price) / order["leverage"]
                    self.balance -= (margin_required + commission)




    def _open_position(self, direction, price_offset, leverage):
        length_orders = len(self.limit_order_list)
        limit_price = self.current_price * (1 + price_offset)
        position_value = self.bet * leverage
        size = position_value / limit_price

        new_order = {
            "type": direction,
            "entry_price": limit_price if direction <= 1 else self.current_price,
            "size": size,
            "leverage": leverage,
            "age": self.total_age + 1,
            "activated": 1 if direction > 4 else 0  # маркет-ордера сразу активны
        }


        if length_orders == 0: # просто добавляем нвоый
            self.limit_order_list.append(new_order)
            if direction in (5, 6):
                self.balance -= self.bet / leverage



        elif length_orders == 1:
            old_dir = self.limit_order_list[0]['type']
            if (old_dir in [0, 5] and direction in [1, 6]) or (old_dir in [1, 6] and direction in [0, 5]): #хочет зайти против себя
                self._exit_market()
                self.total_age = 0
            elif old_dir == direction: # даливаем
                existing_order = self.limit_order_list[0]
                current_size = existing_order["size"]
                new_size = current_size * 1.5
                added_size = new_size - current_size

                existing_order["size"] = new_size
                existing_order["leverage"] = leverage

                if direction in [5, 6]:  # маркет
                    added_entry_price = self.current_price
                    added_value = added_size * added_entry_price
                    margin_required = added_value / leverage
                    self.balance -= margin_required

                else:  # лимит
                    added_entry_price = self.current_price * (1 + price_offset)

                existing_order["entry_price"] = round(
                        (existing_order["entry_price"] * current_size + added_entry_price * added_size) / new_size, 2
                )
            else: # открываем новый
                if direction <= 1:
                    # Для лимитных ордеров – создаем заявку с лимитной ценой.
                    new_order["entry_price"] = limit_price
                    new_order["activated"] = 0
                    self.limit_order_list.append(new_order)
                else:
                    # Для маркет-ордера – сразу активируем и списываем стоимость.
                    new_order["entry_price"] = self.current_price
                    new_order["activated"] = 1
                    position_value = new_order["size"] * self.current_price
                    margin_required = position_value / new_order["leverage"]
                    self.balance -= margin_required
                    self.limit_order_list.append(new_order)


        else:
            old_dir = self.limit_order_list[0]['type']
            if (old_dir == 0 and direction in (1, 6)) or (old_dir == 1 and direction in (0, 5)): # хочет зайти против себя
                self._exit_market()
            elif direction == 0:
                self.add_to_position(0, leverage, price_offset)
            elif direction == 1:
                self.add_to_position(0, leverage, price_offset)
            elif direction == 5: # маркет лонг
                self.add_to_position(1, leverage)
            else:  # direction == 6, маркет-шорт
                self.add_to_position(1, leverage)



    def add_to_position(self, direction, leverage, price_offset=0):
        if direction == 1:
            existing_order = self.limit_order_list[direction]

            current_size = existing_order["size"]
            new_size = current_size * 1.5
            added_size = new_size - current_size

            existing_order["size"] = new_size
            existing_order["entry_price"] = round(
                    (existing_order["entry_price"] * current_size + self.current_price * added_size) / new_size, 2)
            existing_order["leverage"] = leverage

            added_value = added_size * self.current_price
            margin_required = added_value / leverage
            self.balance -= margin_required

        else:
            existing_order = self.limit_order_list[direction]
            current_size = existing_order["size"]
            new_size = current_size * 1.5
            added_size = new_size - current_size
            new_entry_price = round(
                (existing_order["entry_price"] * current_size + (
                            self.current_price * (1 + price_offset)) * added_size) / new_size, 2)

            existing_order["size"] = new_size
            existing_order["entry_price"] = new_entry_price
            existing_order["leverage"] = leverage


        self.limit_order_list.sort(key=lambda order: order["activated"])


    def _exit_limit(self, price_offset):
        pass
        length = len(self.limit_order_list)
        if length == 2:
            position = self.limit_order_list[0]
            direction = position["type"]
            size = position["size"]
            leverage = position["leverage"]

            if direction == 5:  # лонг → хотим продать выше → offset +
                exit_price = self.current_price * (1 + price_offset)
                exit_type = 1  # лимитный шорт
            elif direction == 6:  # шорт → хотим купить ниже → offset -
                exit_price = self.current_price * (1 - price_offset)
                exit_type = 0  # лимитный лонг



    def _exit_market(self):
        length = len(self.limit_order_list)
        if length == 2:
            self.limit_order_list.pop(0)
            order = self.limit_order_list[0]
            if order["type"] == 5:  # лонг
                profit = (self.current_price - order["entry_price"]) * order["size"]
            else:  # шорт
                profit = (order["entry_price"] - self.current_price) * order["size"]

            commission = self.current_price * order["size"] * self.taker_commission
            self.balance += profit - commission
            self.limit_order_list.pop(0)
        elif length == 1:
            order = self.limit_order_list[0]
            if order["activated"] == 1:
                if order["type"] == 5:
                    profit = (self.current_price - order["entry_price"]) * order["size"]
                else:
                    profit = (order["entry_price"] - self.current_price) * order["size"]
                commission = self.current_price * order["size"] * self.taker_commission
                self.balance += profit - commission

            self.limit_order_list.pop(0)
        self.limit_order_list = []


if __name__ == '__main__':

    df = pd.read_pickle('test_df_for_rl.pkl')
    df = df.drop(['turnover', 'timestamp'], axis=1)
    df_train = df.iloc[:9000].reset_index(drop=True)
    df_test = df.iloc[9000:10000].reset_index(drop=True)
    #print(df.isnull().sum())

    env = DummyVecEnv([lambda: TrainTradingEnv(df_train)])

    model = PPO("MultiInputPolicy", env, verbose=1)  # MlpPolicy
    model.learn(total_timesteps=10_000)  # можешь поставить больше

    #model.save("test_pro_ethmodel")
    print("✅ Обучение завершено и модель сохранена.")

    env_test = TrainTradingEnv(df_test)
    obs, _ = env_test.reset()

    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_test.step(action)
        total_reward += reward
        done = terminated or truncated

    # Финальные результаты
    print(f"📊 Финальный баланс: {env_test.balance:.2f}")
    print(f"💸 Общая награда: {total_reward:.2f}")





