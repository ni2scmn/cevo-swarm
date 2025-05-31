import random
import sys
from itertools import combinations
from pathlib import Path

from scipy.spatial.distance import cdist

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from simulator.lib.metrics import distance_to_closest_ap, fraction_inside_radius, symmetry

from . import Warehouse


# Implements swarm with cultural evolution
class CA(Warehouse):
    PHASE_SOCIAL_LEARNING = 0
    PHASE_UPDATE_BEHAVIOUR = 1
    PHASE_EXECUTE_BEHAVIOUR = 2

    def __init__(
        self,
        width,
        height,
        number_of_boxes,
        box_radius,
        swarm,
        init_object_positions=Warehouse.RANDOM_OBJ_POS,
        box_type_ratio=[1],
        phase_ratio=[0.3, 0.3, 0.4],
        phase_change_rate=10,
        influence_r=100,
        adaptive_rate_tuning=False,
    ):
        super().__init__(
            width,
            height,
            number_of_boxes,
            box_radius,
            swarm,
            init_object_positions=init_object_positions,
            box_type_ratio=box_type_ratio,
        )

        self.influence_r = influence_r
        self.phase_ratio = phase_ratio
        self.social_transmission = []
        self.self_updates = []
        self.r_phase = np.array([])
        self.phase_change_rate = 10  # phase_change_rate
        self.verbose = True
        self.continuous_traits = ["P_m", "D_m", "SC", "r0"]
        self.adapt_rate = adaptive_rate_tuning

    # def update_hook(self):
    #

    def select_phase(self):
        if self.counter % self.phase_change_rate == 0:
            # Define probabilities for each phase (ensure they sum to 1)
            probabilities = self.phase_ratio  # % chance for phase 1, % for phase 2, % for phase 3

            # Generate phase array based on probabilities
            phase = np.random.choice(
                [
                    self.PHASE_SOCIAL_LEARNING,
                    self.PHASE_UPDATE_BEHAVIOUR,
                    self.PHASE_EXECUTE_BEHAVIOUR,
                ],
                size=self.swarm.number_of_agents,
                p=probabilities,
            )
            self.r_phase = phase
        else:
            phase = self.r_phase

        s = np.argwhere(phase == self.PHASE_SOCIAL_LEARNING).flatten()
        u = np.argwhere(phase == self.PHASE_UPDATE_BEHAVIOUR).flatten()
        e = np.argwhere(phase == self.PHASE_EXECUTE_BEHAVIOUR).flatten()
        return s, u, e

    # TODO avoid repetition from warehouse class
    def execute_pickup_dropoff(self, robots):
        self.swarm.nn_pickup_dropoff(self, robots)

        # self.swarm.pickup_box(self, robots)
        # drop = self.swarm.dropoff_box(self, robots)
        # drop = []

        # if len(drop):
        #     # rob_n = self.robot_carrier[drop] # robot IDs to drop boxes
        #     valid_drop = []
        #     rob_n = []
        #     for d in drop:
        #         box_d = cdist([self.box_c[d]], self.box_c).flatten()
        #         count = len(np.argwhere(box_d < 10).flatten())
        #         if count < 3:
        #             valid_drop.append(d)
        #             rob_n.append(self.robot_carrier[d])

        #     self.box_is_free[valid_drop] = 1  # mark boxes as free again
        #     self.swarm.agent_has_box[rob_n] = 0  # mark robots as free again
        #     self.swarm.agent_box_id[rob_n] = -1

    # TODO avoid repetition from warehouse class (post hook)
    def iterate(self, heading_bias=False, box_attraction=False):
        self.rob_d = self.swarm.iterate(
            self.rob_c,
            self.box_c,
            self.box_radius,
            self.box_is_free,
            self.map,
            heading_bias,
            box_attraction,
        )  # the robots move using the random walk function which generates a new deviation (rob_d)

        # handles logic to move boxes with robots/drop boxes
        t = self.counter % 10
        self.rob_c_prev[t] = self.rob_c  # Save a record of centre coordinates before update
        self.rob_c = self.rob_c + self.rob_d  # robots centres change as they move
        active_boxes = self.box_is_free == 0  # boxes which are on a robot
        self.box_d = (
            np.array((active_boxes, active_boxes)).T * self.rob_d[self.robot_carrier]
        )  # move the boxes by the amount equal to the robot carrying them
        self.box_c = self.box_c + self.box_d

        self.swarm.compute_metrics(self)
        s, u, e = self.select_phase()
        self.socialize(s)
        self.update(u)
        self.execute_pickup_dropoff(e)

        # metric_ap_dist = distance_to_closest_ap(
        #     self.box_c, np.asarray(self.ap)
        # )
        # print(metric_ap_dist)
        # metric_inside_r_10 = fraction_inside_radius(
        #     self.box_c, np.asarray(self.ap), 10
        # )
        # metric_inside_r_100 = fraction_inside_radius(
        #     self.box_c, np.asarray(self.ap), 100
        # )
        # metric_symmetry_x = symmetry(self.box_c, (self.width, self.height), "x_axis")
        # metric_symmetry_y = symmetry(self.box_c, (self.width, self.height), "y_axis")
        # metric_symmetry_d1 = symmetry(
        #     self.box_c, (self.width, self.height), "diagonal1"
        # )
        # metric_symmetry_d2 = symmetry(
        #     self.box_c, (self.width, self.height), "diagonal2"
        # )

        if (
            self.adapt_rate
            and self.counter > self.swarm.mem_size
            and self.counter % self.swarm.mem_size == 0
        ):
            self.adaptive_rate_tuning()

        self.counter += 1
        self.swarm.counter = self.counter

    def socialize(self, agent_ids):
        used = set()
        noise_strength = 0.01  # Adjust based on your scale
        self.social_transmission = []

        for id1, id2 in combinations(agent_ids, 2):
            if id1 in used or id2 in used:
                continue

            dist = self.swarm.agent_dist[id1][id2]
            if dist >= self.influence_r:
                continue

            # Get influence rates
            rate1 = self.swarm.influence_rate[id1]
            rate2 = self.swarm.influence_rate[id2]

            # Determine influencee and influencer
            if rate1 > rate2:
                influencer, influencee = id1, id2
                influence_prob = rate1 * (1 - rate2)
                reverse_influence_prob = rate2 * (1 - rate1)
            elif rate2 > rate1:
                influencer, influencee = id2, id1
                influence_prob = rate2 * (1 - rate1)
                reverse_influence_prob = rate1 * (1 - rate2)
            else:
                influencer, influencee = id1, id2  # Arbitrary order
                influence_prob = rate1 * (1 - rate2)
                reverse_influence_prob = influence_prob  # Same value
                # or
                # continue  # no update if influence is identical

            weight = influence_prob
            rev_weight = reverse_influence_prob

            if self.verbose:
                print(
                    f"Agents {influencer} (more influential) & {influencee} interacting — influence_prob: {influence_prob:.2f}, dist: {dist:.2f}"
                )
                used.update([id1, id2])

            self.social_transmission.append([id1, id2])

            influencer_params = self.swarm.agents[influencer][0].belief_space.get_weights()
            influencee_params = self.swarm.agents[influencee][0].belief_space.get_weights()

            # TODO: as update mask not updating every parameter?
            if random.random() < influence_prob:
                influencee_params = (
                    influencee_params
                    + weight * (influencer_params - influencee_params)
                    + np.random.normal(0, noise_strength, influencee_params.shape)
                )
                self.swarm.agents[influencee][0].belief_space.set_weights(influencee_params)

            if random.random() < reverse_influence_prob:
                influencer_params = (
                    influencer_params
                    - rev_weight * (influencer_params - influencee_params)
                    + np.random.normal(0, noise_strength, influencer_params.shape)
                )
                self.swarm.agents[influencer][0].belief_space.set_weights(influencer_params)

    # TODO asynchronous evo ?
    # This is called after the main step function (step forward in swarm behaviour)
    def update(self, agent_ids):
        self.self_updates = agent_ids
        noise_strength = 0.01  # Small amount of stochasticity

        for rob_id in agent_ids:
            belief_space_weights = self.swarm.agents[rob_id][0].belief_space.get_weights()
            rob_nn_weights = self.swarm.agents[rob_id][0].control_network.get_weights()

            weight = 1 - self.swarm.resistance_rate[rob_id]

            rob_nn_weights = (
                rob_nn_weights
                + weight * (belief_space_weights - rob_nn_weights)
                + np.random.normal(0, noise_strength, rob_nn_weights.shape)
            )
            self.swarm.agents[rob_id][0].control_network.set_weights(rob_nn_weights)

    def adaptive_rate_tuning(self, alpha_inf=0.05, alpha_res=-1):
        """
        Updates each agent's rates based on novelty.
        * eta_alpha :  between 0 and 1 ,
        Learning Rate controls how quickly the influence rate changes based on novelty
        * gamma_alpha : between -1 and 1
        Sensitivity parameter controls direction and magnitude of the update,

        """

        for agent_id in range(self.swarm.number_of_agents):
            cur_inf_r = self.swarm.influence_rate[agent_id]
            cur_res_r = self.swarm.resistance_rate[agent_id]

            # Update rule
            new_inf_r = cur_inf_r + alpha_inf * self.swarm.novelty_behav[agent_id]
            new_res_r = cur_res_r + alpha_res * self.swarm.novelty_env[agent_id]

            # Clamp to [0, 1]
            self.swarm.influence_rate[agent_id] = max(0.0, min(1.0, new_inf_r))
            self.swarm.resistance_rate[agent_id] = max(0.0, min(1.0, new_res_r))
