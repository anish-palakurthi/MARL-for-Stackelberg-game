import numpy as np
import math
import MATest.utils as util
import MATest.State_Transition_AutoRegressive_AF as ENV_AR_AF
import MATest.DDPG_PER_leader as ddpg_leader
import MATest.DDPG_PER_follower as ddpg_follower

total_relay = 1
total_action = 20
GAMMA = 0.9

def multiagent_ddpg_alliance():
    from MATest.State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num, beta
    total_relay = relay_num
    train_outer_loop = 100
    test_outer_loop = 100
    traj_len = 1000
    ddpg_source = ddpg_follower.DDPG_PER()
    ddpg_relay_league = ddpg_leader.DDPG_PER()
    MEMORY_CAPACITY = 10000
    var = 3
    train_flag = True
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source, result_source_in_train = [], []
    result_relay, result_relay_in_train = [], []
    for episode_out in range(train_outer_loop+test_outer_loop):
        if episode_out >= train_outer_loop:
            train_flag = False
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        state_slot_t0 = channel_gains_t0
        state_relay_t0 = [item for item in state_slot_t0]
        state_source_t0 = [item for item in state_slot_t0]
        state_source_t0.append(0)
        state_source_t0.append(0)
        reward_in_traj_relay, reward_in_traj_source = 0, 0
        for episode in range(traj_len):
            a_r = ddpg_relay_league.choose_action(np.array(state_relay_t0))
            a_r_0 = np.clip(np.random.normal(a_r[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6)
            relay_selection = math.ceil((a_r_0 + total_relay) / 2.0)
            a_r_1 = np.clip(np.random.normal(a_r[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6)
            price_for_power = (a_r_1 + cost_limitation_per_watt) / 2.0
            state_source_t0[-2]=relay_selection
            state_source_t0[-1]=price_for_power
            a_s = ddpg_source.choose_action(np.array(state_source_t0))
            a_s_0 = np.clip(np.random.normal(a_s[0], var)*power_relay_max, -power_relay_max+1e-6, power_relay_max-1e-6)
            power_amount = (a_s_0 + power_relay_max) / 2.0
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            state_slot_t1 = channel_gains_t1
            reward_source, reward_relay, reward_source_STE, reward_relay_STE\
                = env.step(state_t1=state_slot_t0,
                               state_t2=state_slot_t1,
                               relay=relay_selection,
                               cost_relay=price_for_power,
                               power_r=power_amount,
                               power_s=power_source_max,
                               alpha=alpha,
                               bandwidth=1.0)
            if ddpg_source.pointer >= MEMORY_CAPACITY-1 and ddpg_relay_league.pointer >= MEMORY_CAPACITY-1 and train_flag:
                var *= .9995
                ddpg_source.learn()
                ddpg_relay_league.learn()
            state_relay_t1 = [item for item in state_slot_t1]
            a_r_next = ddpg_relay_league.choose_action(np.array(state_relay_t1))
            relay_selection_next = math.ceil((np.clip(np.random.normal(a_r_next[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6) + total_relay) / 2.0)
            price_for_power_next = (np.clip(np.random.normal(a_r_next[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6) + cost_limitation_per_watt) / 2.0
            state_source_t1 = [item for item in state_slot_t1]
            state_source_t1.append(relay_selection_next)
            state_source_t1.append(price_for_power_next)
            ddpg_relay_league.store_transition(
                s=np.array(state_relay_t0),
                a=a_r,
                r=reward_relay,
                s_=state_relay_t1
            )
            ddpg_source.store_transition(
                s=np.array(state_source_t0),
                a=a_s,
                r=reward_source,
                s_=state_source_t1
            )
            state_slot_t0 = state_slot_t1
            state_source_t0 = state_source_t1
            state_relay_t0 = state_relay_t1
            reward_in_traj_relay = reward_in_traj_relay + reward_relay
            reward_in_traj_source = reward_in_traj_source + reward_source
            base_epis, base_perc1, base_perc2 = 60, 0.95, 0.9  # optional, sometimes helpful
            if episode_out > base_epis and train_flag and reward_source / reward_source_STE > base_perc1 and reward_relay / reward_relay_STE > base_perc2:
                train_flag = False
        if episode_out < train_outer_loop:
            result_source_in_train.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay_in_train.append(reward_in_traj_relay / 1.0 / traj_len)
        if episode_out >= train_outer_loop:
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)


def leader_learning_follower_gaming():
    from MATest.State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num, beta
    total_relay = relay_num
    train_outer_loop = 100
    test_outer_loop = 100
    traj_len = 1000
    ddpg_relay_league = ddpg_leader.DDPG_PER()
    MEMORY_CAPACITY = 10000
    var = 3
    train_flag = True
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source, result_source_in_train = [], []
    result_relay, result_relay_in_train = [], []
    for episode_out in range(train_outer_loop+test_outer_loop):
        if episode_out >= train_outer_loop:
            train_flag = False
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        state_slot_t0 = channel_gains_t0
        state_relay_t0 = [item for item in state_slot_t0]
        reward_in_traj_relay, reward_in_traj_source = 0, 0
        for episode in range(traj_len):
            a_r = ddpg_relay_league.choose_action(np.array(state_relay_t0))
            a_r_0 = np.clip(np.random.normal(a_r[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6)
            relay_selection = math.ceil((a_r_0 + total_relay) / 2.0)
            a_r_1 = np.clip(np.random.normal(a_r[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6)
            price_for_power = (a_r_1 + cost_limitation_per_watt) / 2.0
            price_list = [1e3] * total_relay
            price_list[int(relay_selection - 1)] = price_for_power
            SG_solutions_source, SG_rewards_source, relay_STE_index = \
                env.find_best_response_to_prices(
                    state_t1=state_slot_t0,
                    state_t2=state_slot_t0,
                    power_s=power_source_max,
                    alpha=alpha,
                    price_list=price_list,
                    bandwidth=1.0
                )
            power_amount = SG_solutions_source[relay_selection-1]
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            state_slot_t1 = channel_gains_t1
            reward_source, reward_relay, _, _\
                = env.step(state_t1=state_slot_t0,
                               state_t2=state_slot_t1,
                               relay=relay_selection,
                               cost_relay=price_for_power,
                               power_r=power_amount,
                               power_s=power_source_max,
                               alpha=alpha,
                               bandwidth=1.0)
            state_relay_t1 = [item for item in state_slot_t1]
            ddpg_relay_league.store_transition(
                s=np.array(state_relay_t0),
                a=a_r,
                r=reward_relay,
                s_=state_relay_t1
            )
            state_slot_t0 = state_slot_t1
            state_relay_t0 = state_relay_t1
            reward_in_traj_relay = reward_in_traj_relay + reward_relay
            reward_in_traj_source = reward_in_traj_source + reward_source
            if ddpg_relay_league.pointer >= MEMORY_CAPACITY-1 and train_flag:
                var *= .9995
                ddpg_relay_league.learn()
        if episode_out < train_outer_loop:
            result_source_in_train.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay_in_train.append(reward_in_traj_relay / 1.0 / traj_len)
        if episode_out >= train_outer_loop:
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)


def distri_relays_cooperative_partial():
    from MATest.State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num
    total_relay = relay_num
    train_outer_loop = 100
    test_outer_loop_1 = 100
    traj_len = 1000
    ddpg_source = ddpg_follower.DDPG_PER()
    ddpg_relay_list = []
    for i in range(total_relay):
        agent_relay_i = ddpg_leader.DDPG_PER(a_dim=2, s_dim=3, a_bound=[1.0, 1.0])
        ddpg_relay_list.append(agent_relay_i)
    MEMORY_CAPACITY = 10000
    var = 3.0
    train_flag = True
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source, result_source_in_train = [], []
    result_relay, result_relay_in_train = [], []
    for episode_out in range(train_outer_loop+test_outer_loop_1):
        if episode_out >= train_outer_loop and episode_out < (train_outer_loop+test_outer_loop_1):
            train_flag = False
            var = 0
        env.reset_channels()
        if True:
            channel_gains_t0 = env.get_state()
            state_slot_t0 = channel_gains_t0
            state_relay_t0 = [item for item in state_slot_t0]
            state_source_t0 = [item for item in state_slot_t0]
            state_source_t0.append(0)
            state_source_t0.append(0)
            reward_in_traj_relay, reward_in_traj_source = 0, 0
            for episode in range(traj_len):
                action_relays_list = []
                action_relay_index_list = []
                action_recommend_price_list = []
                for relay_agent_index in range(total_relay):
                    a_r = ddpg_relay_list[relay_agent_index].choose_action(np.array([
                        state_relay_t0[relay_agent_index],
                        state_relay_t0[relay_agent_index+total_relay],
                        state_relay_t0[-1]
                    ]))
                    action_relays_list.append(a_r)
                    a_r_0 = np.clip(np.random.normal(a_r[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6)
                    relay_selection_temp = math.ceil((a_r_0 + total_relay) / 2.0)
                    action_relay_index_list.append(relay_selection_temp)
                    a_r_1 = np.clip(np.random.normal(a_r[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6)
                    price_for_power_temp = (a_r_1 + cost_limitation_per_watt) / 2.0
                    action_recommend_price_list.append(price_for_power_temp)
                relay_selection = util.max_list(action_relay_index_list)
                price_calc, cnt = 0, 0
                for (relay_selection_temp, price_for_power_temp) in zip(action_relay_index_list, action_recommend_price_list):
                    if relay_selection_temp == relay_selection:
                        price_calc = price_calc + price_for_power_temp
                        cnt = cnt + 1
                price_for_power = price_calc / cnt
                state_source_t0[-2]=relay_selection
                state_source_t0[-1]=price_for_power
                a_s = ddpg_source.choose_action(np.array(state_source_t0))
                a_s_0 = np.clip(np.random.normal(a_s[0], var)*power_relay_max, -power_relay_max+1e-6, power_relay_max-1e-6)
                power_amount = (a_s_0 + power_relay_max) / 2.0
                env.generate_next_state()
                channel_gains_t1 = env.get_state()
                state_slot_t1 = channel_gains_t1
                reward_source, reward_relay, reward_source_STE, reward_relay_STE\
                    = env.step(state_t1=state_slot_t0,
                                   state_t2=state_slot_t1,
                                   relay=relay_selection,
                                   cost_relay=price_for_power,
                                   power_r=power_amount,
                                   power_s=power_source_max,
                                   alpha=alpha,
                                   bandwidth=1.0)
                if ddpg_source.pointer >= MEMORY_CAPACITY-1 and ddpg_source.pointer >= MEMORY_CAPACITY-1 and train_flag:
                    var *= .9995
                    if var < 0.1:
                        var = 0.1
                    ddpg_source.learn()
                    for relay_agent_index in range(total_relay):
                        ddpg_relay_list[relay_agent_index].learn()
                state_relay_t1 = [item for item in state_slot_t1]
                if True:
                    action_relay_index_list_next = []
                    action_recommend_price_list_next = []
                    for relay_agent_index in range(total_relay):
                        a_r_next = ddpg_relay_list[relay_agent_index].choose_action(np.array([
                            state_relay_t1[relay_agent_index],
                            state_relay_t1[relay_agent_index + total_relay],
                            state_relay_t1[-1]
                        ]))
                        relay_selection_temp_next = math.ceil((np.clip(np.random.normal(a_r_next[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6) + total_relay) / 2.0)
                        price_for_power_temp_next = (np.clip(np.random.normal(a_r_next[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6) + cost_limitation_per_watt) / 2.0
                        action_relay_index_list_next.append(relay_selection_temp_next)
                        action_recommend_price_list_next.append(price_for_power_temp_next)
                    relay_selection_next = util.max_list(action_relay_index_list_next)
                    price_calc, cnt = 0, 0
                    for (relay_selection_temp_next, price_for_power_temp_next) in zip(action_relay_index_list_next, action_recommend_price_list_next):
                        if relay_selection_temp_next == relay_selection_next:
                            price_calc = price_calc + price_for_power_temp_next
                            cnt = cnt + 1
                    price_for_power_next = price_calc / cnt
                state_source_t1 = [item for item in state_slot_t1]
                state_source_t1.append(relay_selection_next)
                state_source_t1.append(price_for_power_next)
                if train_flag:
                    for relay_agent_index in range(total_relay):
                        ddpg_relay_list[relay_agent_index].store_transition(
                            s=np.array(state_relay_t0),
                            a=action_relays_list[relay_agent_index],
                            r=reward_relay,
                            s_=state_relay_t1
                        )
                    ddpg_source.store_transition(
                        s=np.array(state_source_t0),
                        a=a_s,
                        r=reward_source,
                        s_=state_source_t1
                    )
                state_slot_t0 = state_slot_t1
                state_source_t0 = state_source_t1
                state_relay_t0 = state_relay_t1
                reward_in_traj_relay = reward_in_traj_relay + reward_relay
                reward_in_traj_source = reward_in_traj_source + reward_source
        if episode_out < train_outer_loop:
            result_source_in_train.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay_in_train.append(reward_in_traj_relay / 1.0 / traj_len)
        if episode_out >= train_outer_loop and episode_out < (train_outer_loop + test_outer_loop_1):
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)



# multiagent_ddpg_alliance()

# leader_learning_follower_gaming()

# distri_relays_cooperative_partial()

