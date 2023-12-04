from CQL_agent import ContinuousCQL as CQL



agent = CQL(state_dim=15, action_dim=2, hidden_dim=64, target_entropy=-2,
                            q_n_hidden_layers=1, max_action=1, qf_lr=3e-4, policy_lr=6e-5,device="cuda:0")
agent.load("./models/cql_dataset_sac_best.pt")
agent.actor.eval()