import numpy as np
import math
import State_Transition_AutoRegressive_AF as ENV_AR_AF
import DDPG_single as ddpg_single

def SG_SingleDDPG():
    weight_param = 1.5
    from State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num
    total_relay = relay_num
    train_outer_loop = 100
    test_outer_loop = 100
    traj_len = 1000
    ddpg_agent= ddpg_single.DDPG_PER()
    MEMORY_CAPACITY = 10000
    var = 3
    train_flag = True
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source, result_source_in_train = [], []
    result_relay, result_relay_in_train = [], []
    result_single, result_single_in_train = [], []
    for episode_out in range(train_outer_loop+test_outer_loop):
        print(f"Episode {episode_out}/{train_outer_loop+test_outer_loop-1}")
        if episode_out >= train_outer_loop:
            train_flag = False
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        state_slot_t0 = channel_gains_t0
        state_t0 = [item for item in state_slot_t0]
        reward_in_traj_relay, reward_in_traj_source = 0, 0
        reward_in_traj_single = 0
        for episode in range(traj_len):
            actions = ddpg_agent.choose_action(np.array(state_t0))
            a_relay = np.clip(np.random.normal(actions[0], var) * total_relay, -total_relay + 1e-6, total_relay - 1e-6)
            relay_selection = math.ceil((a_relay + total_relay) / 2.0)
            a_cost = np.clip(np.random.normal(actions[1], var) * cost_limitation_per_watt, -cost_limitation_per_watt + 1e-6, cost_limitation_per_watt - 1e-6)
            price_for_power = (a_cost + cost_limitation_per_watt) / 2.0
            a_power = np.clip(np.random.normal(actions[2], var)*power_relay_max, -power_relay_max+1e-6, power_relay_max-1e-6)
            power_amount = (a_power + power_relay_max) / 2.0
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
            reward_temp = reward_source + weight_param * reward_relay
            if ddpg_agent.pointer >= MEMORY_CAPACITY-1 and train_flag:
                var *= .9995
                ddpg_agent.learn()
            state_t1 = [item for item in state_slot_t1]
            ddpg_agent.store_transition(
                s=np.array(state_t0),
                a=actions,
                r=reward_temp,
                s_=state_t1
            )
            state_slot_t0 = state_slot_t1
            reward_in_traj_relay = reward_in_traj_relay + reward_relay
            reward_in_traj_source = reward_in_traj_source + reward_source
            reward_in_traj_single = reward_in_traj_single + reward_temp
        if episode_out < train_outer_loop:
            result_source_in_train.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay_in_train.append(reward_in_traj_relay / 1.0 / traj_len)
            result_single_in_train.append(reward_in_traj_single / 1.0 / traj_len)
        if episode_out >= train_outer_loop:
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)
            result_single.append(reward_in_traj_single / 1.0 / traj_len)
    print("SG_SingleDDPG results:")
    print(f"Source in train: {np.array(result_source_in_train)}")
    print(f"Relay in train: {np.array(result_relay_in_train)}")
    print(f"Single in train: {np.array(result_single_in_train)}")
    print(f"Source in test: {np.array(result_source)}")
    print(f"Relay in test: {np.array(result_relay)}")
    print(f"Single in test: {np.array(result_single)}")
    return np.array(result_source_in_train), np.array(result_relay_in_train), np.array(result_single_in_train), np.array(result_source), np.array(result_relay), np.array(result_single)

result_source_in_train, result_relay_in_train, result_single_in_train, result_source, result_relay, result_single = SG_SingleDDPG()