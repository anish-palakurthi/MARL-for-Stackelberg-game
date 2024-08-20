import numpy as np
import utils as util
import State_Transition_AutoRegressive_AF as ENV_AR_AF

total_relay = 5
total_action = 20
GAMMA = 0.9

def Random_AF():
    from State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num, beta
    total_relay = relay_num
    train_outer_loop = 0
    test_outer_loop = 100
    traj_len = 1000
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source = []
    result_relay = []
    for episode_out in range(train_outer_loop+test_outer_loop):
        print(f"Episode {episode_out}/{train_outer_loop+test_outer_loop-1}")
        channel_gains_t0 = env.get_state()
        state_slot_t0 = channel_gains_t0
        reward_in_traj_relay, reward_in_traj_source = 0, 0
        import random
        for episode in range(traj_len):
            action_relays = []
            for relay_agent_index in range(total_relay):
                a_r = random.random() * cost_limitation_per_watt
                action_relays.append(a_r)
            a_s = util.random_selection(env.relay_list, power_relay_max)
            source_action_relay = a_s[0]
            source_action_power = a_s[1]
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            state_slot_t1 = channel_gains_t1
            reward_source, reward_relay, reward_source_STE, reward_relay_STE\
                = env.step(state_t1=state_slot_t0,
                               state_t2=state_slot_t1,
                               relay=source_action_relay,
                               cost_relay=action_relays[int(source_action_relay)-1],
                               power_r=source_action_power,
                               power_s=power_source_max,
                               alpha=alpha,
                               bandwidth=1.0)
            state_slot_t0 = state_slot_t1
            reward_in_traj_relay = reward_in_traj_relay + reward_relay
            reward_in_traj_source = reward_in_traj_source + reward_source
        if episode_out >= train_outer_loop:
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)
    print("Random_AF results:")
    print(f"Source: {np.array(result_source)}")
    print(f"Relay: {np.array(result_relay)}")
    return np.array(result_source), np.array(result_relay)

def Pure_Game_Theory_Method_instan():
    from State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num
    total_relay = relay_num
    test_outer_loop = 100
    traj_len = 1000
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source_STE = []
    result_relay_STE = []
    for episode_out in range(test_outer_loop):
        print(f"Episode {episode_out}/{test_outer_loop-1}")
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        reward_in_traj_relay_STE, reward_in_traj_source_STE = 0, 0
        for episode in range(traj_len):
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            action_relays, SG_rewards_relays, SG_solutions_source, SG_rewards_source, relay_STE_index = \
                env.find_SG_solutions(state_t1=channel_gains_t0,
                                      state_t2=channel_gains_t1,
                                      power_s=power_source_max,
                                      alpha=alpha,
                                      bandwidth=1.0)
            reward_relay_STE = SG_rewards_relays[relay_STE_index]
            reward_source_STE = SG_rewards_source[relay_STE_index]
            channel_gains_t0 = channel_gains_t1
            reward_in_traj_relay_STE = reward_in_traj_relay_STE + reward_relay_STE
            reward_in_traj_source_STE = reward_in_traj_source_STE + reward_source_STE
        result_source_STE.append(reward_in_traj_source_STE / 1.0 / traj_len)
        result_relay_STE.append(reward_in_traj_relay_STE / 1.0 / traj_len)
    result_capacity_STE = []
    for (source_STE, relay_STE) in zip(result_source_STE, result_relay_STE):
        capacity_STE = source_STE + relay_STE
        result_capacity_STE.append(capacity_STE)
    print("Pure_Game_Theory_Method_instan results:")
    print(f"Source: {np.array(result_source_STE)}")
    print(f"Relay: {np.array(result_relay_STE)}")
    print(f"Capacity: {np.array(result_capacity_STE)}")
    return np.array(result_source_STE), np.array(result_relay_STE), np.array(result_capacity_STE)

def Pure_Game_Theory_Method_instan_competitive_relays():
    from State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num
    total_relay = relay_num
    test_outer_loop = 100
    traj_len = 1000
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source_STE = []
    result_relay_STE = []
    for episode_out in range(test_outer_loop):
        print(f"Episode {episode_out}/{test_outer_loop-1}")
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        reward_in_traj_relay_STE, reward_in_traj_source_STE = 0, 0
        for episode in range(traj_len):
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            Cost_k_STE_clip, reward_relay_STE, Power_k_STE_clip, reward_source_STE, winner_relay_index \
                = env.find_SG_solutions_competitive_relays(
                state_t1=channel_gains_t0,
                state_t2=channel_gains_t1,
                power_s=power_source_max,
                alpha=alpha,
            )
            channel_gains_t0 = channel_gains_t1
            reward_in_traj_relay_STE = reward_in_traj_relay_STE + reward_relay_STE
            reward_in_traj_source_STE = reward_in_traj_source_STE + reward_source_STE
        result_source_STE.append(reward_in_traj_source_STE / 1.0 / traj_len)
        result_relay_STE.append(reward_in_traj_relay_STE / 1.0 / traj_len)
    result_capacity_STE = []
    for (source_STE, relay_STE) in zip(result_source_STE, result_relay_STE):
        capacity_STE = source_STE + relay_STE
        result_capacity_STE.append(capacity_STE)
    print("Pure_Game_Theory_Method_instan_competitive_relays results:")
    print(f"Source: {np.array(result_source_STE)}")
    print(f"Relay: {np.array(result_relay_STE)}")
    print(f"Capacity: {np.array(result_capacity_STE)}")
    return np.array(result_source_STE), np.array(result_relay_STE), np.array(result_capacity_STE)

result_source_1, result_relay_1 = Random_AF()
result_source_2, result_relay_2, result_capacity_2 = Pure_Game_Theory_Method_instan()
# result_source_3, result_relay_3, result_capacity_3 = Pure_Game_Theory_Method_instan_competitive_relays()