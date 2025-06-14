from copy import deepcopy
from operator import ne
from pathlib import Path
import re
import sys
from tqdm import tqdm
import concurrent.futures
from unittest import result
from simulator.lib.metrics import distance_to_closest_ap, symmetry
from simulator.wh_sim.neuroevolution import one_point_crossover, point_mutate

from simulator.lib import TrainSaveTo

dir_root = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import random
import threading
import os
from os.path import dirname, realpath
import datetime
import time
import json

from . import Swarm, CA, Warehouse, Robot

from .nn import FeedforwardNN, NNBeliefSpace, random_weight_init, sigmoid, softmax


def eval_entity(entity, warehouse, swarm, cfg):
    for i in range(cfg.get("warehouse", "number_of_agents")):
        swarm.agents[i][0].control_network.set_weights(entity[0])
    warehouse.swarm = swarm  # Update the swarm in the warehouse

    entity_log = {
        "box_c": {},
        "rob_c": {},
    }

    while warehouse.counter <= cfg.get("time_limit"):
        warehouse.iterate(cfg.get("heading_bias"), cfg.get("box_attraction"))
        
        entity_log["box_c"][warehouse.counter] = warehouse.box_c.tolist()
        entity_log["rob_c"][warehouse.counter] = warehouse.rob_c.tolist()

    fitness_fun = set_pretrain_metric(cfg.get("train", "metric"))
    fitness = fitness_fun(warehouse.box_c, np.asarray(warehouse.ap), ((warehouse.width, warehouse.height)))

    return (fitness, entity_log)

    # # if agent has picked up boxes, return the negative distance to the closest AP
    # if np.sum(warehouse.agent_box_pickup_count) == 0:
    #     # If no boxes were picked up, return a large negative value
    #     return (-10000, entity_log)
    # elif np.sum(warehouse.agent_box_dropoff_count) == 0:
    #     # If no boxes were dropped off, punish the fitness
    #     return (fitness * 2 if fitness < 0 else fitness / 2, entity_log)
    # else:
    #     # If boxes were picked up, calculate the fitness
    #     # Calculate the fitness as the negative distance to the closest AP
    #     return (fitness, entity_log)


def export_entity_logs(entity_logs, ts, idx_gen, idx_entity):
    try:
        st = TrainSaveTo(ts)
        _ = st.export_data("pretrain", idx_gen, entity_logs["box_c"], "box_c_" + str(idx_entity))
        _ = st.export_data("pretrain", idx_gen, entity_logs["rob_c"], "rob_c_" + str(idx_entity))
    except Exception as e:
        print(f"Error exporting entity logs for entity {idx_entity}: {e}")
        pass        


def parse_population_csv(population_csv_path):
    """
    Parse the population CSV file and return a list of numpy arrays.
    """
    population = []
    with open(population_csv_path, 'r') as f:
        f.readline()  # Skip the header line
        for line in f:
            # Convert each value to float and wrap in a NumPy array
            float_array = np.array([float(value) for value in line.strip().split(',')], dtype=float)
            population.append(float_array)
    return population

class Pretrain:
    def __init__(
        self,
        config,
        verbose=False,  # verbosity
        random_seed=None,
    ):
        self.cfg = config
        self.verbose = verbose
        self.exit_threads = False

        self.population_size = self.cfg.get("train", "population_size")
        self.n_generations = self.cfg.get("train", "n_generations")
        self.metric = self.cfg.get("train", "metric")
        self.mutation_rate = self.cfg.get("train", "mutation_rate")
        self.crossover_rate = self.cfg.get("train", "crossover_rate")
        self.elitism = self.cfg.get("train", "elitism")

        self.swarm = self.init_swarm()
        self.warehouse = self.init_warehouse()
        self.population = self.init_population()

        self.log_data = {}
        self.ts = int(datetime.datetime.now().timestamp())
        self.st = TrainSaveTo(self.ts)


    def init_warehouse(self):
        warehouse = CA(
            self.cfg.get("warehouse", "width"),
            self.cfg.get("warehouse", "height"),
            self.cfg.get("warehouse", "number_of_boxes"),
            self.cfg.get("warehouse", "box_radius"),
            self.swarm,
            self.cfg.get("warehouse", "object_position"),
            self.cfg.get("box_type_ratio"),
            self.cfg.get("phase_ratio"),
            self.cfg.get("phase_change_rate"),
            self.cfg.get("influence_r"),
            self.cfg.get("adaptive_rate_tuning"),
        )

        warehouse.generate_ap(self.cfg)
        warehouse.verbose = self.verbose
        return warehouse

    def init_swarm(self):
        swarm = Swarm(
            repulsion_o=self.cfg.get("warehouse", "repulsion_object"),
            repulsion_w=self.cfg.get("warehouse", "repulsion_wall"),
            heading_change_rate=self.cfg.get("heading_change_rate"),
        )

        nn_layers = self.cfg.get("nn_controller", "nn_layers")
        weight_init = self.cfg.get("nn_controller", "weight_init")
        weight_init_fun = lambda: random.uniform(int(weight_init[0]), int(weight_init[1]))

        activation = self.cfg.get("nn_controller", "activation_funcs")
        activation_funcs = []

        for af in activation:
            if af == "sigmoid":
                activation_funcs.append(sigmoid)
            elif af == "softmax":
                activation_funcs.append(softmax)
            else:
                raise ValueError("Unknown activation function")

        control_network = FeedforwardNN(
            layers=nn_layers,
            weight_init=weight_init_fun,
            activation_fun=activation_funcs,
        )
        belief_space = NNBeliefSpace(
            bs_nn_weights=np.random.uniform(-1, 1, size=control_network.get_weights().shape)
        )
        robot_obj = Robot(
            self.cfg.get("robot", "radius"),
            self.cfg.get("robot", "max_v"),
            camera_sensor_range=self.cfg.get("robot", "camera_sensor_range"),
            control_network=control_network,
            belief_space=belief_space,
        )
        for _ in range(self.cfg.get("warehouse", "number_of_agents")):
            swarm.add_agents(robot_obj, 1)

        swarm.generate()
        swarm.init_params(self.cfg)
        return swarm

    def init_population(self):
        population = []

        if self.cfg.get("train", "start_from") is not None:
            population = parse_population_csv(self.cfg.get("train", "start_from"))
            population = [(np.array(weights), -1e6) for weights in population] # (weights, fitness)
            return population

        weight_init = self.cfg.get("nn_controller", "weight_init")
        for _ in range(self.population_size):
            nn_weights = np.random.uniform(
                int(weight_init[0]),
                int(weight_init[1]),
                size=self.swarm.agents[0][0].control_network.get_weights().shape,
            )
            population.append((nn_weights, -1e6))  # (weights, fitness)
        return population

    def eval_generation(self, idx_gen):
        args = []
        self.log_data[idx_gen] = {
            "population": [],
            "fitness": [],
            # "box_c": [],
            # "rob_c": [],
        }
        for i in range(self.population_size):
            entity = self.population[i]
            warehouse = deepcopy(self.warehouse)  # Copy warehouse to avoid modifying the original
            swarm = deepcopy(self.swarm)  # Copy swarm to avoid modifying the original
            args.append((entity, warehouse, swarm, self.cfg))

        # Use multithreading to evaluate entities in parallel
        if self.cfg.get("train", "parallel"):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(eval_entity, *zip(*args)),
                        total=len(args),
                        desc="Evaluating entities",
                    )
                )
        else:
            results = []
            for arg in tqdm(args, desc="Evaluating entities"):
                results.append(eval_entity(*arg))

        entity_logs = []

        for i, r in enumerate(results):
            fitness, entity_log = r
            print(f"\tEntity {i + 1} fitness: {fitness}")
            self.population[i] = (self.population[i][0], fitness)  # Update fitness
            # Log data for this generation
            self.log_data[idx_gen]["population"].append(self.population[i][0])
            self.log_data[idx_gen]["fitness"].append(fitness)
            #entity_logs.append(entity_log)

            self.st.gen_save_dirname("pretrain", str(self.ts) + "_train_" + str(idx_gen), makedir=True)

        # if self.cfg.get("train", "parallel"):
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        #         ts = deepcopy(self.ts)
        #         results = list(
        #             tqdm(
        #                 executor.map(export_entity_logs, entity_logs, [ts] * len(entity_logs), [idx_gen] * len(entity_logs), range(self.population_size)),
        #                 total=len(entity_logs),
        #                 desc="Exporting entity logs",
        #             )
        #         )
        # else:
        #     raise NotImplementedError("Sequential export not implemented yet")

        _ = self.st.export_data(
            "pretrain", idx_gen, self.log_data[idx_gen]["population"], "population"
        )
        _ = self.st.export_data(
            "pretrain", idx_gen, self.log_data[idx_gen]["fitness"], "fitness"
        )


    def run_episode(self):
        while self.warehouse.counter <= self.cfg.get("time_limit"):
            self.iterate()

    def iterate(self):
        self.warehouse.iterate(self.cfg.get("heading_bias"), self.cfg.get("box_attraction"))
        counter = self.warehouse.counter
        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter % 100 == 0:
                print("=", end="", flush=True)

    def get_fitness(self):
        return -distance_to_closest_ap(
            self.warehouse.box_c,
            np.asarray(self.warehouse.ap),
        )

    def run(self) -> None:
        for idx_gen in range(self.n_generations):
            print(f"Generation {idx_gen + 1}/{self.n_generations}")
            self.eval_generation(idx_gen)
            print("Best fitness:", max([entity[1] for entity in self.population]))
            print("Average fitness:", np.mean([entity[1] for entity in self.population]))
            self.population = self.evolve_population()

    def evolve_population(self):
        new_population = []
        sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)

        # Elitism: keep the best individuals
        if self.elitism:
            n_elite = max(1, int(self.elitism * self.population_size))
            new_population.extend(sorted_population[:n_elite])

        while len(new_population) < self.population_size:
            parent1 = self.select_parent_tournament(sorted_population, tournament_size=6)
            parent2 = self.select_parent_tournament(sorted_population, tournament_size=6)
            if np.random.rand() < self.crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            child1 = point_mutate(
                child1, self.mutation_rate, mutation=np.random.normal(0, 1, size=child1[0].shape)
            )
            child2 = point_mutate(
                child2, self.mutation_rate, mutation=np.random.normal(0, 1, size=child2[0].shape)
            )
            new_population.append((child1, -1e6))
            new_population.append((child2, -1e6))
        # Ensure the new population size matches the original
        new_population = new_population[: self.population_size]

        return new_population

    def select_parent(self, population):
        # Select a parent based on fitness (roulette wheel selection)
        total_fitness = sum(entity[1] for entity in population)
        selection_probs = [entity[1] / total_fitness for entity in population]
        idx = np.random.choice(len(population), p=selection_probs)
        return population[idx][0]

    def select_parent_tournament(self, population, tournament_size=3):
        # Select a parent using tournament selection
        tournament = random.sample(population, tournament_size)
        best_entity = max(tournament, key=lambda x: x[1])
        return best_entity[0]


def set_pretrain_metric(metric_str: str):
    if metric_str == "ap_distance":
        return lambda box_c, ap_c, _dim: -distance_to_closest_ap(box_c, ap_c)
    elif metric_str == "left_right":
        return lambda box_c, _ap_c, dim: symmetry(box_c, dim, "x_axis")
    else:
        raise ValueError("unvalid pretrain metric")
