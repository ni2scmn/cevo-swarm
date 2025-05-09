from pathlib import Path
import sys
import random
from itertools import combinations
from scipy.spatial.distance import cdist

dir_root = Path(__file__).resolve().parents[1]

import numpy as np

from . import Warehouse


# Implements swarm with cultural evolution
class CA(Warehouse):

    PHASE_SOCIAL_LEARNING = 0
    PHASE_UPDATE_BEHAVIOUR = 1
    PHASE_EXECUTE_BEHAVIOUR = 2

    def __init__(self, width, height, number_of_boxes, box_radius, swarm,
		init_object_positions=Warehouse.RANDOM_OBJ_POS, 
        box_type_ratio=[1], phase_ratio=[0.3,0.3,0.4], phase_change_rate=10, influence_r=100):
        super().__init__(width, height, number_of_boxes, box_radius, swarm,
		    init_object_positions=init_object_positions, box_type_ratio=box_type_ratio)
        
        self.influence_r = influence_r
        self.phase_ratio = phase_ratio
        self.social_transmission =[]
        self.self_updates = []
        self.r_phase = np.array([])
        self.phase_change_rate = phase_change_rate
        self.verbose = True
        self.continuous_traits = ['P_m', 'D_m', 'SC', 'r0']


    # def update_hook(self):
        #

    def select_phase(self):
        if self.counter % self.phase_change_rate == 0:
            # Define probabilities for each phase (ensure they sum to 1)
            probabilities = self.phase_ratio  # % chance for phase 1, % for phase 2, % for phase 3
            
            # Generate phase array based on probabilities
            phase = np.random.choice([self.PHASE_SOCIAL_LEARNING, self.PHASE_UPDATE_BEHAVIOUR, self.PHASE_EXECUTE_BEHAVIOUR],
                                    size=self.swarm.number_of_agents,
                                    p=probabilities)
            self.r_phase = phase
        else:
            phase = self.r_phase
        
        s = np.argwhere(phase==self.PHASE_SOCIAL_LEARNING).flatten()
        u = np.argwhere(phase==self.PHASE_UPDATE_BEHAVIOUR).flatten()
        e = np.argwhere(phase==self.PHASE_EXECUTE_BEHAVIOUR).flatten()
        return s,u,e

    # TODO avoid repetition from warehouse class
    def execute_pickup_dropoff(self, robots):
        self.swarm.pickup_box(self, robots)
        drop = self.swarm.dropoff_box(self, robots)
		
        if len(drop):
            # rob_n = self.robot_carrier[drop] # robot IDs to drop boxes
            valid_drop = []
            rob_n = []
            for d in drop:
                box_d = cdist([self.box_c[d]],self.box_c).flatten()
                count = len(np.argwhere(box_d<10).flatten())
                if count < 3:
                    valid_drop.append(d)
                    rob_n.append(self.robot_carrier[d])

            self.box_is_free[valid_drop] = 1 # mark boxes as free again
            self.swarm.agent_has_box[rob_n] = 0 # mark robots as free again
            self.swarm.agent_box_id[rob_n] = -1

    # TODO avoid repetition from warehouse class (post hook)
    def iterate(self, heading_bias=False, box_attraction=False):     
        self.rob_d = self.swarm.iterate(
			self.rob_c, 
			self.box_c, 
			self.box_radius,
			self.box_is_free, 
			self.map, 
			heading_bias,
			box_attraction) # the robots move using the random walk function which generates a new deviation (rob_d)
        
		# handles logic to move boxes with robots/drop boxes
        t = self.counter%10
        self.rob_c_prev[t] = self.rob_c # Save a record of centre coordinates before update
        self.rob_c = self.rob_c + self.rob_d # robots centres change as they move
        active_boxes = self.box_is_free == 0 # boxes which are on a robot
        self.box_d = np.array((active_boxes,active_boxes)).T*self.rob_d[self.robot_carrier] # move the boxes by the amount equal to the robot carrying them 
        self.box_c = self.box_c + self.box_d
		
        self.swarm.compute_metrics()
        s,u,e = self.select_phase()   
        self.socialize(s)
        self.update(u)
        self.execute_pickup_dropoff(e)

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
                    f"Agents {influencer} (more influential) & {influencee} interacting — influence_prob: {influence_prob:.2f}, dist: {dist:.2f}")
                used.update([id1, id2])

            self.social_transmission.append([id1, id2])

            # Each param: behaviour → BS_ version
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                source_array = getattr(self.swarm, attr) # Behaviour param
                target_array = getattr(self.swarm, f'BS_{attr}')  # belief space param

                param_size = self.swarm.no_ap if attr in ['P_m', 'D_m'] else self.swarm.no_box_t

                start_inf = influencer * param_size
                start_infce = influencee * param_size

                for i in range(param_size):
                    if attr in self.continuous_traits :
                        v_inf = source_array[start_inf + i]
                        v_infce = source_array[start_infce + i]
                        if random.random() < influence_prob:
                            new_value = v_infce + weight * (v_inf - v_infce) + random.gauss(0,noise_strength)
                            target_array[start_infce + i] = min(max(new_value, 0), 1)
                        if random.random() < reverse_influence_prob:
                            new_value = v_inf - rev_weight * (v_inf - v_infce) + random.gauss(0,noise_strength)
                            target_array[start_inf + i] = min(max(new_value, 0), 1)
                    else:
                        if random.random() < influence_prob:
                            target_array[start_infce + i] = source_array[start_inf + i]
                        if random.random() < reverse_influence_prob:
                            target_array[start_inf + i] = source_array[start_infce + i]






                # After the update, store the modified target_array back to self.BS_
                setattr(self.swarm, f'BS_{attr}', target_array)



    # TODO asynchronous evo ?
    # This is called after the main step function (step forward in swarm behaviour)
    def update(self, agent_ids):

        self.self_updates = agent_ids
        noise_strength = 0.01  # Small amount of stochasticity

        for id in agent_ids:
            # Each param: behaviour → BS_ version
            for attr in ['P_m', 'D_m', 'SC', 'r0']:
                target_array = getattr(self.swarm, attr)  # Behaviour param
                source_array= getattr(self.swarm, f'BS_{attr}')  # belief space param

                param_size = self.swarm.no_ap if attr in ['P_m', 'D_m'] else self.swarm.no_box_t

                start_index = id * param_size
                weight = 1 - self.swarm.resistance_rate[id]

                for i in range(param_size):
                    v_behavior = target_array[start_index + i]
                    v_belief = source_array[start_index + i]

                    if attr in self.continuous_traits :
                        # Gradual update for continuous traits with noise
                        new_value = (
                                v_behavior + weight * (v_belief - v_behavior) + random.gauss(0, noise_strength)
                        )
                        target_array[start_index + i] = min(max(new_value, 0), 1)
                    else:
                        # Probabilistic full copy for discrete traits
                        if random.random() < weight:  # Use weight as probability for the update
                            target_array[start_index + i] = v_belief  # Full adoption of belief


                # After the update, store the modified target_array back to self.BS_
                setattr(self.swarm, attr, target_array)


