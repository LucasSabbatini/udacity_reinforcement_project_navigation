import time

def test_agent(agent, env, brain_name, times=5, max_t=500, freq=60): 

    scores = []
    for i in range(times):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]         
        score = 0                                       
        for i in range(max_t):
            action = agent.act(state, eps=0)
            env_info = env.step(action.astype(np.int32))[brain_name] 
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]                
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
            time.sleep(1/freq)
        
        scores.append(score)
        print(f"Episode ended with score: {score}. Starting a new one")
        time.sleep(2)
        
    return scores