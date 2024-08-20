import random
import numpy as np
import ChannelMDP as chmdp

power_relay_max = 2.0
cost_limitation_per_watt = 10.0
power_source_max = 1.0
alpha = 0.1
beta = 0.1
relay_num = 5
matrix_locations_distance_r2s = [80, 90, 70, 110, 85]
matrix_locations_distance_r2d = [95, 105, 110, 100, 100]
locations_distance_s2d = 150

class State_Transition():
    def __init__(self, noise_power_settings=1e-9):
        self.relay_list = [i + 1 for i in range(relay_num)]
        self.noise_power = noise_power_settings
        self.ch_s1 = chmdp.Channel(dis=100)
        self.ch_s2 = chmdp.Channel(dis=100)
        self.ch_s3 = chmdp.Channel(dis=100)
        self.ch_s4 = chmdp.Channel(dis=100)
        self.ch_s5 = chmdp.Channel(dis=100)
        self.ch_1d = chmdp.Channel(dis=100)
        self.ch_2d = chmdp.Channel(dis=100)
        self.ch_3d = chmdp.Channel(dis=100)
        self.ch_4d = chmdp.Channel(dis=100)
        self.ch_5d = chmdp.Channel(dis=100)
        self.ch_sd = chmdp.Channel(dis=locations_distance_s2d)
        self.ch_list = [self.ch_s1, self.ch_s2, self.ch_s3, self.ch_s4, self.ch_s5,
                        self.ch_1d, self.ch_2d, self.ch_3d, self.ch_4d, self.ch_5d,
                        self.ch_sd
                        ]

    def reset_channels(self):
        self.ch_s1 = chmdp.Channel(dis=matrix_locations_distance_r2s[0])
        self.ch_s2 = chmdp.Channel(dis=matrix_locations_distance_r2s[1])
        self.ch_s3 = chmdp.Channel(dis=matrix_locations_distance_r2s[2])
        self.ch_s4 = chmdp.Channel(dis=matrix_locations_distance_r2s[3])
        self.ch_s5 = chmdp.Channel(dis=matrix_locations_distance_r2s[4])
        self.ch_1d = chmdp.Channel(dis=matrix_locations_distance_r2d[0])
        self.ch_2d = chmdp.Channel(dis=matrix_locations_distance_r2d[1])
        self.ch_3d = chmdp.Channel(dis=matrix_locations_distance_r2d[2])
        self.ch_4d = chmdp.Channel(dis=matrix_locations_distance_r2d[3])
        self.ch_5d = chmdp.Channel(dis=matrix_locations_distance_r2d[4])
        self.ch_sd = chmdp.Channel(dis=locations_distance_s2d)
        self.ch_list = [self.ch_s1, self.ch_s2, self.ch_s3, self.ch_s4, self.ch_s5,
                        self.ch_1d, self.ch_2d, self.ch_3d, self.ch_4d, self.ch_5d,
                        self.ch_sd
                        ]

    def generate_next_state(self):
        for channel in self.ch_list:
            channel.sampleCh()

    def get_state(self):
        channels_gain = []
        for channel in self.ch_list:
            channels_gain.append(channel.calcChannelGain())
        return channels_gain

    def step(self, state_t1, state_t2, relay=1, power_r=0, cost_relay=0.2, power_s=0.1, alpha=2, bandwidth=1.0):
        s_r_gain_in_1st_hop = state_t1[relay - 1]
        s_d_gain_in_1st_hop = state_t1[2 * relay_num]
        SNR_sr = power_s * s_r_gain_in_1st_hop / 1.0 / self.noise_power
        SNR_sd = power_s * s_d_gain_in_1st_hop / 1.0 / self.noise_power
        d_r_gain_in_2nd_hop = state_t2[relay - 1 + relay_num]
        SNR_rd = power_r * d_r_gain_in_2nd_hop / 1.0 / self.noise_power
        r_s_rate = 0.5 * np.log2(1 + SNR_sd + SNR_sr*SNR_rd/1.0/(SNR_sr+SNR_rd+1))
        r_s_cost = alpha * cost_relay*power_r
        r_r_cost = beta * cost_relay*power_r
        reward_source = r_s_rate - r_s_cost
        reward_relay = r_r_cost
        reward_source_STE_max = 0
        reward_relay_STE_max = 0
        relay_STE_index = 0
        for relay_index in range(relay_num):  # 每个relay与source形成一个斯塔克伯格博弈
            s_r_gain_in_1st_hop_STE = state_t1[relay_index]
            s_d_gain_in_1st_hop_STE = state_t1[2 * relay_num]
            SNR_sr_STE = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            SNR_sd_STE = power_s * s_d_gain_in_1st_hop_STE / 1.0 / self.noise_power
            d_r_gain_in_2nd_hop_STE = state_t2[relay_index + relay_num]
            A_k = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            B_k = (power_s * s_r_gain_in_1st_hop_STE + self.noise_power) / 1.0 / d_r_gain_in_2nd_hop_STE  # G_k
            C_k = np.log(2) * np.power(A_k, 2) * np.power(B_k, 2) * np.power(alpha, 2)  # A_k
            D_k = (2 * np.power(A_k, 2) + 2 * A_k) * B_k * alpha  # B_k
            E_k = (np.log(2) * A_k + 2 * np.log(2)) * B_k * alpha  # D_k

            cost_k_list_temp = []
            cost_k_list_temp.append(1e-12)
            cost_k_list_temp.append(cost_limitation_per_watt)
            Cost_k_STE_frac_root1 = D_k * E_k * np.sqrt(np.power(E_k,2) - np.log(2) * C_k) - D_k * np.power(E_k,2) + np.log(2) * C_k * D_k
            Cost_k_STE_frac_root2 = - D_k * E_k * np.sqrt(np.power(E_k,2) - np.log(2) * C_k) - D_k * np.power(E_k,2) + np.log(2) * C_k * D_k
            Cost_k_STE_frac_2 = 2 * C_k * np.power(E_k,2) - 2 * np.log(2) * np.power(C_k,2)
            Cost_k_STE = max(Cost_k_STE_frac_root1 / 1.0 / Cost_k_STE_frac_2, Cost_k_STE_frac_root2 / 1.0 / Cost_k_STE_frac_2)
            if (Cost_k_STE > 1e-12)and (Cost_k_STE < cost_limitation_per_watt):
                cost_k_list_temp.append(Cost_k_STE)
            const_1 = np.power(np.log(2) * A_k * B_k, 2) / np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2)
            const_2 = 2 * A_k * (A_k + 1) * B_k / (np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2) * alpha)
            const_3 = (np.log(2) * A_k + 2 * np.log(2)) * B_k / (2 * np.log(2) * (A_k + 1))
            C0 = const_2 / (np.power(power_relay_max + const_3, 2) - const_1)
            if (C0 > 1e-12) and (C0 < cost_limitation_per_watt):
                cost_k_list_temp.append(C0)
            C1 = const_2 / (np.power(0 + const_3, 2) - const_1)

            power_k_list_temp = []
            for item in cost_k_list_temp:
                power_k_temp = (np.sqrt(np.log(2)) * np.sqrt(C_k * np.power(item,2) + D_k * item) - E_k * item) / (2 * np.log(2) * (A_k + 1) * alpha * item)
                power_k_temp_clip = np.clip(power_k_temp, 0, power_relay_max)
                power_k_list_temp.append(power_k_temp_clip)

            reward_relay_k_list_temp = []
            for (cost_temp, power_temp) in zip(cost_k_list_temp, power_k_list_temp):
                reward_relay_k_temp = beta * cost_temp * power_temp
                reward_relay_k_list_temp.append(reward_relay_k_temp)
            index_of_best_solution = reward_relay_k_list_temp.index(max(reward_relay_k_list_temp))
            Cost_k_STE_clip = cost_k_list_temp[index_of_best_solution]
            Power_k_STE_clip = power_k_list_temp[index_of_best_solution]

            SNR_rd_STE = Power_k_STE_clip * d_r_gain_in_2nd_hop_STE / 1.0 / self.noise_power
            r_s_rate_STE = 0.5 * np.log2(1 + SNR_sd_STE + SNR_sr_STE*SNR_rd_STE/1.0/(SNR_sr_STE+SNR_rd_STE+1))
            r_s_cost_STE = alpha * Cost_k_STE_clip * Power_k_STE_clip
            reward_source_STE = r_s_rate_STE - r_s_cost_STE
            r_r_cost_STE = beta * Cost_k_STE_clip * Power_k_STE_clip
            reward_relay_STE = r_r_cost_STE
            if reward_relay_STE > reward_relay_STE_max:
                reward_source_STE_max = reward_source_STE
                reward_relay_STE_max = reward_relay_STE
                relay_STE_index = relay_index
        return reward_source, reward_relay, reward_source_STE_max, reward_relay_STE_max

    def find_SG_solutions(self,  state_t1, state_t2, power_s=0.1, alpha=2, bandwidth=1.0):
        reward_source_STE_max = 0
        reward_relay_STE_max = 0
        relay_STE_index = 0
        SG_solutions_source = []
        SG_solutions_relays = []
        SG_rewards_source = []
        SG_rewards_relays = []
        for relay_index in range(relay_num):  # 每个relay与source形成一个斯塔克伯格博弈
            s_r_gain_in_1st_hop_STE = state_t1[relay_index]
            s_d_gain_in_1st_hop_STE = state_t1[2 * relay_num]
            SNR_sr_STE = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            SNR_sd_STE = power_s * s_d_gain_in_1st_hop_STE / 1.0 / self.noise_power
            d_r_gain_in_2nd_hop_STE = state_t2[relay_index + relay_num]
            A_k = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            B_k = (power_s * s_r_gain_in_1st_hop_STE + self.noise_power) / 1.0 / d_r_gain_in_2nd_hop_STE
            C_k = np.log(2) * np.power(A_k, 2) * np.power(B_k, 2) * np.power(alpha, 2)
            D_k = (2 * np.power(A_k, 2) + 2 * A_k) * B_k * alpha
            E_k = (np.log(2) * A_k + 2 * np.log(2)) * B_k * alpha

            cost_k_list_temp = []
            cost_k_list_temp.append(1e-12)
            cost_k_list_temp.append(cost_limitation_per_watt)
            Cost_k_STE_frac_root1 = D_k * E_k * np.sqrt(np.power(E_k, 2) - np.log(2) * C_k) - D_k * np.power(E_k, 2) + np.log(2) * C_k * D_k
            Cost_k_STE_frac_root2 = - D_k * E_k * np.sqrt(np.power(E_k, 2) - np.log(2) * C_k) - D_k * np.power(E_k, 2) + np.log(2) * C_k * D_k
            Cost_k_STE_frac_2 = 2 * C_k * np.power(E_k, 2) - 2 * np.log(2) * np.power(C_k, 2)
            Cost_k_STE = max(Cost_k_STE_frac_root1 / 1.0 / Cost_k_STE_frac_2, Cost_k_STE_frac_root2 / 1.0 / Cost_k_STE_frac_2)
            if Cost_k_STE > 1e-12 and Cost_k_STE < cost_limitation_per_watt:
                cost_k_list_temp.append(Cost_k_STE)
            const_1 = np.power(np.log(2) * A_k * B_k, 2) / np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2)
            const_2 = 2 * A_k * (A_k + 1) * B_k / (np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2) * alpha)
            const_3 = (np.log(2) * A_k + 2 * np.log(2)) * B_k / (2 * np.log(2) * (A_k + 1))
            C0 = const_2 / (np.power(power_relay_max + const_3, 2) - const_1)
            if C0 > 1e-12 and C0 < cost_limitation_per_watt:
                cost_k_list_temp.append(C0)
            C1 = const_2 / (np.power(0 + const_3, 2) - const_1)

            power_k_list_temp = []
            for item in cost_k_list_temp:
                power_k_temp = (np.sqrt(np.log(2)) * np.sqrt(C_k * np.power(item, 2) + D_k * item) - E_k * item) / (2 * np.log(2) * (A_k + 1) * alpha * item)
                power_k_temp_clip = np.clip(power_k_temp, 1e-12, power_relay_max)
                power_k_list_temp.append(power_k_temp_clip)

            reward_relay_k_list_temp = []
            for (cost_temp, power_temp) in zip(cost_k_list_temp, power_k_list_temp):
                reward_relay_k_temp = beta * cost_temp * power_temp
                reward_relay_k_list_temp.append(reward_relay_k_temp)
            index_of_best_solution = reward_relay_k_list_temp.index(max(reward_relay_k_list_temp))
            Cost_k_STE_clip = cost_k_list_temp[index_of_best_solution]
            Power_k_STE_clip = power_k_list_temp[index_of_best_solution]

            SNR_rd_STE = Power_k_STE_clip * d_r_gain_in_2nd_hop_STE / 1.0 / self.noise_power
            r_s_rate_STE = 0.5 * np.log2(1 + SNR_sd_STE + SNR_sr_STE * SNR_rd_STE / 1.0 / (SNR_sr_STE + SNR_rd_STE + 1))
            r_s_cost_STE = alpha * Cost_k_STE_clip * Power_k_STE_clip
            reward_source_STE = r_s_rate_STE - r_s_cost_STE
            r_r_cost_STE = beta * Cost_k_STE_clip * Power_k_STE_clip
            reward_relay_STE = r_r_cost_STE
            SG_solutions_relays.append(Cost_k_STE_clip)
            SG_rewards_relays.append(reward_relay_STE)
            SG_solutions_source.append(Power_k_STE_clip)
            SG_rewards_source.append(reward_source_STE)
            if reward_relay_STE > reward_relay_STE_max:
                reward_source_STE_max = reward_source_STE
                reward_relay_STE_max = reward_relay_STE
                relay_STE_index = relay_index
        return SG_solutions_relays, SG_rewards_relays, SG_solutions_source, SG_rewards_source, relay_STE_index

    def find_best_response_to_prices(self, state_t1, state_t2, power_s=0.1, alpha=2, price_list=[], bandwidth=1.0):
        reward_source_STE_max = 0
        relay_STE_index = 0
        SG_solutions_source = []
        SG_rewards_source = []
        for relay_index in range(relay_num):  # 每个relay与source形成一个斯塔克伯格博弈
            s_r_gain_in_1st_hop_STE = state_t1[relay_index]
            s_d_gain_in_1st_hop_STE = state_t1[2 * relay_num]
            SNR_sr_STE = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            SNR_sd_STE = power_s * s_d_gain_in_1st_hop_STE / 1.0 / self.noise_power
            d_r_gain_in_2nd_hop_STE = state_t2[relay_index + relay_num]
            A_k = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
            B_k = (power_s * s_r_gain_in_1st_hop_STE + self.noise_power) / 1.0 / d_r_gain_in_2nd_hop_STE
            C_k = np.log(2) * np.power(A_k, 2) * np.power(B_k, 2) * np.power(alpha, 2)
            D_k = (2 * np.power(A_k, 2) + 2 * A_k) * B_k * alpha
            E_k = (np.log(2) * A_k + 2 * np.log(2)) * B_k * alpha

            Cost_k_STE_clip = price_list[relay_index]
            Power_k_STE = (np.sqrt(np.log(2)) * np.sqrt(
                C_k * np.power(Cost_k_STE_clip, 2) + D_k * Cost_k_STE_clip) - E_k * Cost_k_STE_clip) / (2 * np.log(2) * (A_k + 1) * alpha * Cost_k_STE_clip)
            Power_k_STE_clip = np.clip(Power_k_STE, 1e-12, power_relay_max-1e-12)

            SNR_rd_STE = Power_k_STE_clip * d_r_gain_in_2nd_hop_STE / 1.0 / self.noise_power
            r_s_rate_STE = 0.5 * np.log2(
                1 + SNR_sd_STE + SNR_sr_STE * SNR_rd_STE / 1.0 / (SNR_sr_STE + SNR_rd_STE + 1))
            r_s_cost_STE = alpha * Cost_k_STE_clip * Power_k_STE_clip
            reward_source_STE = r_s_rate_STE - r_s_cost_STE
            r_r_cost_STE = beta * Cost_k_STE_clip * Power_k_STE_clip
            SG_solutions_source.append(Power_k_STE_clip)
            SG_rewards_source.append(reward_source_STE)
            if reward_source_STE > reward_source_STE_max:
                reward_source_STE_max = reward_source_STE
                relay_STE_index = relay_index
        return SG_solutions_source, SG_rewards_source, relay_STE_index

    def find_SG_solutions_competitive_relays(self,  state_t1, state_t2, power_s=0.1, alpha=2, bandwidth=1.0):
        # 筛选winner relay
        utility_source_list_under_cmin = []
        for relay_index in range(relay_num):
            s_r_gain_in_1st_hop= state_t1[relay_index]
            s_d_gain_in_1st_hop = state_t1[2 * relay_num]
            SNR_sr = power_s * s_r_gain_in_1st_hop / self.noise_power
            SNR_sd = power_s * s_d_gain_in_1st_hop / self.noise_power
            d_r_gain_in_2nd_hop = state_t2[relay_index + relay_num]
            SNR_rd = power_relay_max * d_r_gain_in_2nd_hop / self.noise_power
            us_temp = 0.5 * np.log2(1 + SNR_sr * SNR_rd / (SNR_sr + SNR_rd + 1))
            utility_source_list_under_cmin.append(us_temp)
        us_k_cmin_max = max(utility_source_list_under_cmin)
        winner_relay_index = utility_source_list_under_cmin.index(us_k_cmin_max)
        # 计算us下界
        utility_source_list_under_cmin.remove(us_k_cmin_max)
        us_tilde = max(utility_source_list_under_cmin)
        # 求解对应的Ck solution
        Cost_k_max_game = 0
        s_r_gain_in_1st_hop_STE = state_t1[winner_relay_index]
        s_d_gain_in_1st_hop_STE = state_t1[2 * relay_num]
        SNR_sr_STE = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power
        SNR_sd_STE = power_s * s_d_gain_in_1st_hop_STE / 1.0 / self.noise_power
        d_r_gain_in_2nd_hop_STE = state_t2[winner_relay_index + relay_num]
        A_k = power_s * s_r_gain_in_1st_hop_STE / 1.0 / self.noise_power  # gamma_sk
        B_k = (power_s * s_r_gain_in_1st_hop_STE + self.noise_power) / 1.0 / d_r_gain_in_2nd_hop_STE  # G_k
        C_k = np.log(2) * np.power(A_k, 2) * np.power(B_k, 2) * np.power(alpha, 2)  # A_k
        D_k = (2 * np.power(A_k, 2) + 2 * A_k) * B_k * alpha  # B_k
        E_k = (np.log(2) * A_k + 2 * np.log(2)) * B_k * alpha  # D_k

        cost_k_list_temp = []
        cost_k_list_temp.append(1e-60)
        cost_k_list_temp.append(cost_limitation_per_watt)
        Cost_k_STE_frac_root1 = D_k * E_k * np.sqrt(np.power(E_k, 2) - np.log(2) * C_k) - D_k * np.power(E_k, 2) + np.log(2) * C_k * D_k
        Cost_k_STE_frac_root2 = - D_k * E_k * np.sqrt(np.power(E_k, 2) - np.log(2) * C_k) - D_k * np.power(E_k, 2) + np.log(2) * C_k * D_k
        Cost_k_STE_frac_2 = 2 * C_k * np.power(E_k, 2) - 2 * np.log(2) * np.power(C_k, 2)
        Cost_k_STE = max(Cost_k_STE_frac_root1 / 1.0 / Cost_k_STE_frac_2, Cost_k_STE_frac_root2 / 1.0 / Cost_k_STE_frac_2)
        if (Cost_k_STE > 1e-12) and (Cost_k_STE < cost_limitation_per_watt):
            cost_k_list_temp.append(Cost_k_STE)
        const_1 = np.power(np.log(2) * A_k * B_k, 2) / np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2)
        const_2 = 2 * A_k * (A_k + 1) * B_k / (np.power(2 * np.log(2) * A_k + 2 * np.log(2), 2) * alpha)
        const_3 = (np.log(2) * A_k + 2 * np.log(2)) * B_k / (2 * np.log(2) * (A_k + 1))
        C0 = const_2 / (np.power(power_relay_max + const_3, 2) - const_1)
        if (C0 > 1e-12) and (C0 < cost_limitation_per_watt):
            cost_k_list_temp.append(C0)
        C1 = const_2 / (np.power(0 + const_3, 2) - const_1)
        if Cost_k_STE < C0:
            Cost_k_max_game = (us_k_cmin_max - us_tilde) / (alpha * power_relay_max)
        else:
            # 方法1：用sympy库解方程 (内存不足)
            # import sympy
            # x = sympy.symbols("x")
            # equation_x = 0.5 * sympy.log(1 + A_k/(1 + B_k/( (const_1+const_2/x)**0.5 - const_3 ))) / sympy.log(2) \
            #     - alpha * x * ((const_1 + const_2 / x)**0.5 - const_3)- us_tilde
            # Cost_k_max_game = sympy.solve(equation_x, x)
            # 方法2：逐次逼近
            delta_cost_k = 1e-4
            cost_k_iteration_temp = 1 * delta_cost_k
            iteration_loops = int(cost_limitation_per_watt / delta_cost_k)
            for i in range(iteration_loops):
                optimal_set_list = []
                for possible_solution in cost_k_list_temp:
                    if possible_solution < cost_k_iteration_temp:
                        optimal_set_list.append(possible_solution)
                optimal_set_list.append(cost_k_iteration_temp)
                power_k_list_temp = []
                for item in optimal_set_list:
                    power_k_temp = (np.sqrt(np.log(2)) * np.sqrt(C_k * np.power(item, 2) + D_k * item) - E_k * item) / (2 * np.log(2) * (A_k + 1) * alpha * item)
                    power_k_temp_clip = np.clip(power_k_temp, 0, power_relay_max)
                    power_k_list_temp.append(power_k_temp_clip)
                reward_source_list_temp = []
                reward_relay_k_list_temp = []
                for (cost_temp, power_temp) in zip(optimal_set_list, power_k_list_temp):
                    SNR_rd_temp = power_temp * d_r_gain_in_2nd_hop_STE / self.noise_power
                    reward_source_temp = 0.5 * np.log2(1 + SNR_sr_STE * SNR_rd_temp / (SNR_sr_STE + SNR_rd_temp + 1)) - alpha * cost_temp * power_temp
                    reward_source_list_temp.append(reward_source_temp)
                    reward_relay_k_temp = beta * cost_temp * power_temp
                    reward_relay_k_list_temp.append(reward_relay_k_temp)
                relay_k_best_choice = reward_relay_k_list_temp.index(max(reward_relay_k_list_temp))
                if reward_source_list_temp[relay_k_best_choice] < us_tilde:
                    break
                else:
                    Cost_k_max_game = cost_k_iteration_temp
                cost_k_iteration_temp = cost_k_iteration_temp + delta_cost_k
        optimal_set_list = [item for item in cost_k_list_temp]
        cost_k_list_temp = []
        for possible_solution in optimal_set_list:
            if possible_solution < Cost_k_max_game:
                cost_k_list_temp.append(possible_solution)
        if (Cost_k_max_game >=0) and (Cost_k_max_game <= cost_limitation_per_watt):
            cost_k_list_temp.append(Cost_k_max_game)

        power_k_list_temp = []
        for item in cost_k_list_temp:
            power_k_temp = (np.sqrt(np.log(2)) * np.sqrt(C_k * np.power(item, 2) + D_k * item) - E_k * item) / (2 * np.log(2) * (A_k + 1) * alpha * item)
            power_k_temp_clip = np.clip(power_k_temp, 0, power_relay_max)
            power_k_list_temp.append(power_k_temp_clip)

        reward_relay_k_list_temp = []
        for (cost_temp, power_temp) in zip(cost_k_list_temp, power_k_list_temp):
            reward_relay_k_temp = beta * cost_temp * power_temp
            reward_relay_k_list_temp.append(reward_relay_k_temp)
        index_of_best_solution = reward_relay_k_list_temp.index(max(reward_relay_k_list_temp))
        Cost_k_STE_clip = cost_k_list_temp[index_of_best_solution]
        Power_k_STE_clip = power_k_list_temp[index_of_best_solution]
        SNR_rd_STE = Power_k_STE_clip * d_r_gain_in_2nd_hop_STE / 1.0 / self.noise_power
        r_s_rate_STE = 0.5 * np.log2(1 + SNR_sd_STE + SNR_sr_STE * SNR_rd_STE / 1.0 / (SNR_sr_STE + SNR_rd_STE + 1))
        r_s_cost_STE = alpha * Cost_k_STE_clip * Power_k_STE_clip
        reward_source_STE = r_s_rate_STE - r_s_cost_STE
        r_r_cost_STE = beta * Cost_k_STE_clip * Power_k_STE_clip
        reward_relay_STE = r_r_cost_STE
        return Cost_k_STE_clip, reward_relay_STE, Power_k_STE_clip, reward_source_STE, winner_relay_index


def random_pick(some_list, probabilities):
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

