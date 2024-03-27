import numpy as np
import MATest.State_Transition_AutoRegressive_AF as ENV_AR_AF
import MATest.DQN_follower as dqn_follower
import MATest.DQN_leader as dqn_leader

total_power_level = 10
total_price_level = 10

def multiagent_dqn_alliance():
    from MATest.State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num, beta
    total_relay = relay_num
    train_outer_loop = 100
    test_outer_loop = 100
    traj_len = 1000
    step = 0
    Memory_Size = 10000
    Batch_Size = 128
    dqn_source = dqn_follower.DeepQNetwork(
        n_actions=total_power_level,
        n_features=2 * total_relay + 1 + 1,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=Memory_Size,
        batch_size=Batch_Size,
    )
    dqn_relay_league = dqn_leader.DeepQNetwork(
        n_actions=total_price_level * total_relay,
        n_features=2 * total_relay + 1,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=200,
        memory_size=Memory_Size,
        batch_size=Batch_Size,
    )
    env = ENV_AR_AF.State_Transition(noise_power_settings=1e-9)
    env.reset_channels()
    result_source, result_source_in_train = [], []
    result_relay, result_relay_in_train = [], []
    for episode_out in range(train_outer_loop+test_outer_loop):
        env.reset_channels()
        channel_gains_t0 = env.get_state()
        state_slot_t0 = channel_gains_t0
        reward_in_traj_relay, reward_in_traj_source = 0, 0
        for episode in range(traj_len):
            step = step + 1
            s_leader = [item for item in state_slot_t0]
            action_league_agent = dqn_relay_league.choose_action(np.array(s_leader))
            action_relay = action_league_agent // total_price_level + 1
            action_price_level = action_league_agent % total_price_level + 1
            action_price = action_price_level * (cost_limitation_per_watt/total_price_level)
            s_follower = [item for item in state_slot_t0]
            s_follower.append(action_price)
            action_power_level = dqn_source.choose_action(np.array(s_follower))
            action_power = action_power_level * (power_relay_max/total_power_level)
            env.generate_next_state()
            channel_gains_t1 = env.get_state()
            state_slot_t1 = channel_gains_t1
            reward_source, reward_relay, _, _\
                = env.step(state_t1=state_slot_t0,
                               state_t2=state_slot_t1,
                               relay=action_relay,
                               cost_relay=action_price,
                               power_r=action_power,
                               power_s=power_source_max,
                               alpha=alpha,
                               bandwidth=1.0)
            state_next_relay = [item for item in state_slot_t1]
            dqn_relay_league.store_transition(
                s=s_leader,
                a=action_league_agent,
                r=reward_relay,
                s_=state_next_relay,
            )
            state_next_source = [item for item in state_slot_t1]
            action_league_agent_next = dqn_relay_league.choose_action(np.array(state_next_source))
            action_price_next = (action_league_agent_next % total_price_level + 1) * (cost_limitation_per_watt/total_price_level)
            state_next_source.append(action_price_next)
            dqn_source.store_transition(
                s=s_follower,
                a=action_power_level,
                r=reward_source,
                s_=state_next_source,
            )
            if (step > 500) and (step % 5 == 0) and episode_out < train_outer_loop:
                dqn_relay_league.learn()
                dqn_source.learn()
            state_slot_t0 = state_slot_t1
            reward_in_traj_relay = reward_in_traj_relay + reward_relay
            reward_in_traj_source = reward_in_traj_source + reward_source
        if episode_out < train_outer_loop:
            result_source_in_train.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay_in_train.append(reward_in_traj_relay / 1.0 / traj_len)
        if episode_out >= train_outer_loop:
            result_source.append(reward_in_traj_source / 1.0 / traj_len)
            result_relay.append(reward_in_traj_relay / 1.0 / traj_len)


# multiagent_dqn_alliance()
