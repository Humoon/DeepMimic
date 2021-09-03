import numpy as np


class TrainingSet:

    def __init__(self, player_num, max_capacity=10240, training_set_type="training_server"):
        self.training_set_type = training_set_type
        self.player_num = player_num
        self.max_capacity = max_capacity
        self.action_mask_list = []
        self.share_feature_list = []
        self.player_feature_list = []
        self.lstm_state_list = []
        self.pre_action_list = []
        if self.training_set_type == "training_server":
            self.state_value_list = []
            self.action_list = []
            self.action_prob_list = []
            self.q_reward_list = []
            self.gae_advantage_list = []
            self.model_update_times_list = []

    def append_instance(self, instance):
        self.action_mask_list.append(instance.action_mask)
        self.share_feature_list.append(instance.share_feature)
        self.player_feature_list.append(instance.player_feature)
        self.lstm_state_list.append(instance.lstm_state)
        self.pre_action_list.append(instance.pre_action)
        if self.training_set_type == "training_server":
            self.state_value_list.append(instance.state_value)
            self.action_list.append(instance.action)
            self.action_prob_list.append(instance.action_prob)
            self.q_reward_list.append(instance.q_reward)
            self.gae_advantage_list.append(instance.gae_advantage)
            self.model_update_times_list.append(instance.model_update_times)

    def clear(self):
        self.action_mask_list = []
        self.share_feature_list = []
        self.player_feature_list = []
        self.lstm_state_list = []
        self.pre_action_list = []
        if self.training_set_type == "training_server":
            self.state_value_list = []
            self.q_reward_list = []
            self.gae_advantage_list = []
            self.action_prob_list = []
            self.action_list = []
            self.model_update_times_list = []

    def len(self):
        return len(self.share_feature_list)

    def fit_max_size(self):
        if self.len() > self.max_capacity:
            keep_index_start = self.len() - self.max_capacity
            self.action_mask_list = self.action_mask_list[keep_index_start:]
            self.share_feature_list = self.share_feature_list[keep_index_start:]
            self.player_feature_list = self.player_feature_list[keep_index_start:]
            self.lstm_state_list = self.lstm_state_list[keep_index_start:]
            self.pre_action_list = self.pre_action_list[keep_index_start:]
            if self.training_set_type == "training_server":
                self.state_value_list = self.state_value_list[keep_index_start:]
                self.q_reward_list = self.q_reward_list[keep_index_start:]
                self.gae_advantage_list = self.gae_advantage_list[keep_index_start:]
                self.action_prob_list = self.action_prob_list[keep_index_start:]
                self.action_list = self.action_list[keep_index_start:]
                self.model_update_times_list = self.model_update_times_list[keep_index_start:]

    def convert2np(self):
        batch_size = self.len() * self.player_num
        slice_dict = {}
        slice_dict["action_mask"] = np.array(self.action_mask_list).reshape(batch_size, -1)
        slice_dict["share_feature"] = np.array(self.share_feature_list).reshape(self.len(), -1)
        slice_dict["player_feature"] = np.array(self.player_feature_list).reshape(batch_size, -1)
        slice_dict["lstm_state"] = np.array(self.lstm_state_list).reshape(batch_size, -1)
        slice_dict["pre_action"] = np.array(self.pre_action_list).reshape(batch_size, -1)
        if self.training_set_type == "training_server":
            slice_dict["state_value"] = np.array(self.state_value_list).reshape(batch_size, -1)
            slice_dict["q_reward"] = np.array(self.q_reward_list).reshape(batch_size, -1)
            slice_dict["gae_advantage"] = np.array(self.gae_advantage_list).reshape(batch_size, -1)
            slice_dict["action"] = np.array(self.action_list).reshape(-1)
            slice_dict["action_prob"] = np.array(self.action_prob_list).reshape(batch_size, -1)
            slice_dict["model_update_times_list"] = np.array(self.model_update_times_list).reshape(batch_size, -1)
        return slice_dict

    def new_slice(self, n_steps):
        batch_size = self.len() * self.player_num
        slice_dict = {}

        slice_dict["action_mask"] = np.array(self.action_mask_list).reshape(batch_size, -1)
        slice_dict["share_feature"] = np.array(self.share_feature_list).reshape(self.len(), -1)
        slice_dict["player_feature"] = np.array(self.player_feature_list).reshape(batch_size, -1)
        lstm_state = np.array(self.lstm_state_list).reshape([self.len() // n_steps, n_steps, self.player_num, -1])[:, 0]
        slice_dict["lstm_state"] = lstm_state.reshape(batch_size // n_steps, -1)
        slice_dict["pre_action"] = np.array(self.pre_action_list).reshape(batch_size, -1)
        slice_dict["state_value"] = np.array(self.state_value_list).reshape(batch_size, -1)
        slice_dict["q_reward"] = np.array(self.q_reward_list).reshape(batch_size, -1)
        slice_dict["gae_advantage"] = np.array(self.gae_advantage_list).reshape(batch_size, -1)
        slice_dict["action"] = np.array(self.action_list).reshape(-1)
        slice_dict["action_prob"] = np.array(self.action_prob_list).reshape(batch_size, -1)
        slice_dict["model_update_times_list"] = np.array(self.model_update_times_list).reshape(batch_size, -1)
        return slice_dict

    def slice(self, index_list, n_steps):
        batch_size = len(index_list) * self.player_num
        slice_dict = {}

        state_value = np.array([self.state_value_list[i] for i in index_list]) # slice
        slice_dict["state_value"] = state_value.reshape(batch_size, -1)
        q_reward = np.array([self.q_reward_list[i] for i in index_list]) # slice
        slice_dict["q_reward"] = q_reward.reshape(batch_size, -1)
        gae_advantage = np.array([self.gae_advantage_list[i] for i in index_list])
        slice_dict["gae_advantage"] = gae_advantage.reshape(batch_size, -1)
        share_feature = np.array([self.share_feature_list[i] for i in index_list])
        slice_dict["share_feature"] = share_feature.reshape(len(index_list), -1)
        player_feature = np.array([self.player_feature_list[i] for i in index_list])
        slice_dict["player_feature"] = player_feature.reshape(batch_size, -1)
        lstm_state = np.array([self.lstm_state_list[i] for i in index_list])
        lstm_state2 = lstm_state.reshape([len(index_list) // n_steps, n_steps, self.player_num, -1])[:, 0]
        slice_dict["lstm_state"] = lstm_state2.reshape(batch_size // n_steps, -1)
        action = np.array([self.action_list[i] for i in index_list])
        slice_dict["action"] = action.reshape(batch_size)
        action_prob = np.array([self.action_prob_list[i] for i in index_list])
        slice_dict["action_prob"] = action_prob.reshape(batch_size, -1)
        action_mask = np.array([self.action_mask_list[i] for i in index_list])
        slice_dict["action_mask"] = action_mask.reshape(batch_size, -1)
        pre_action = np.array([self.pre_action_list[i] for i in index_list])
        slice_dict["pre_action"] = pre_action.reshape(batch_size, -1)

        return slice_dict

    def get_ave_update_times(self):
        return np.average(self.model_update_times_list)
