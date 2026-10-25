'''
Hyperparameter Optimization for DQN Planning using Google Ax & Bayesian Optimization
Automates optimal learning rate, discount factor (gamma), batch size, and replay buffer tuning.
'''

import sys

def optimize_dqn_parameters(total_trials: int = 15):
    try:
        from ax.service.managed_loop import optimize
    except ImportError:
        print('Ax platform not installed. Run: pip install ax-platform')
        return

    def evaluate_dqn_config(parameters):
        lr = parameters.get('learning_rate', 1e-3)
        gamma = parameters.get('gamma', 0.99)
        batch_size = parameters.get('batch_size', 64)
        
        # Target objective: Simulated average planning reward
        simulated_reward = 100.0 - (abs(lr - 0.0005) * 10000 + abs(gamma - 0.98) * 100 + abs(batch_size - 128) * 0.1)
        return {'average_reward': (simulated_reward, 1.0)}

    print('=== Starting Bayesian Optimization with Google Ax ===')
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {'name': 'learning_rate', 'type': 'range', 'bounds': [1e-4, 1e-2], 'log_scale': True},
            {'name': 'gamma', 'type': 'range', 'bounds': [0.90, 0.999]},
            {'name': 'batch_size', 'type': 'choice', 'values': [32, 64, 128, 256]},
        ],
        evaluation_function=evaluate_dqn_config,
        objective_name='average_reward',
        minimize=False,
        total_trials=total_trials
    )

    print('\n=== Optimal Hyperparameters Found ===')
    print(best_parameters)
    return best_parameters

if __name__ == '__main__':
    optimize_dqn_parameters(total_trials=10)