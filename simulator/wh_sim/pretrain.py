from pathlib import Path
import sys

from simulator.lib.metrics import distance_to_closest_ap

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

        self.init_swarm()
        self.init_warehouse()
        self.population = self.init_population()

    def init_warehouse(self):
        self.warehouse = CA(
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

        self.warehouse.generate_ap(self.cfg)
        self.warehouse.verbose = self.verbose

    def init_swarm(self):
        self.swarm = Swarm(
            repulsion_o=self.cfg.get("warehouse", "repulsion_object"),
            repulsion_w=self.cfg.get("warehouse", "repulsion_wall"),
            heading_change_rate=self.cfg.get("heading_change_rate"),
        )

        nn_layers = self.cfg.get("robot", "nn_layers")
        weight_init = self.cfg.get("robot", "weight_init")
        if weight_init == "random":
            weight_init_fun = random_weight_init
        else:
            raise ValueError("Unknown weight init function")
        activation = self.cfg.get("robot", "activation_funcs")
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
            self.swarm.add_agents(robot_obj, 1)

        self.swarm.generate()
        self.swarm.init_params(self.cfg)

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            nn_weights = np.random.uniform(
                -1, 1, size=self.swarm.agents[0][0].control_network.get_weights().shape
            )
            population.append((nn_weights, -1e6))  # (weights, fitness)
        return population

    def eval_generation(self):
        for i in range(self.population_size):
            entity = self.population[i]
            fitness = self.eval_entity(entity)
            print(f"\tEntity {i + 1} fitness: {fitness}")
            self.population[i] = (entity[0], fitness)  # Update fitness

    def eval_entity(self, entity):
        for i in range(self.cfg.get("warehouse", "number_of_agents")):
            self.swarm.agents[i][0].control_network.set_weights(entity[0])

        self.init_warehouse()

        self.run_episode()
        return self.get_fitness()

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

    def run(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}/{self.n_generations}")
            self.eval_generation()
            self.population = self.evolve_population()
            print("Best fitness:", max([entity[1] for entity in self.population]))
            print("Average fitness:", np.mean([entity[1] for entity in self.population]))

    def evolve_population(self):
        new_population = []
        sorted_population = sorted(self.population, key=lambda x: x[1], reverse=True)

        # Elitism: keep the best individuals
        if self.elitism:
            n_elite = max(1, int(self.elitism * self.population_size))
            new_population.extend(sorted_population[:n_elite])

        while len(new_population) < self.population_size:
            parent1 = self.select_parent(sorted_population)
            parent2 = self.select_parent(sorted_population)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        return new_population

    def select_parent(self, population):
        # Select a parent based on fitness (roulette wheel selection)
        total_fitness = sum(entity[1] for entity in population)
        selection_probs = [entity[1] / total_fitness for entity in population]
        idx = np.random.choice(len(population), p=selection_probs)
        return population[idx]

    def crossover(self, parent1, parent2):
        # Simple crossover: average weights of parents
        child_weights = (parent1[0] + parent2[0]) / 2
        return (child_weights, -1e6)

    def mutate(self, entity):
        # Simple mutation: add small random noise to weights
        if np.random.rand() < self.mutation_rate:
            noise = np.random.normal(0, 0.1, size=entity[0].shape)
            entity = (entity[0] + noise, entity[1])  # Keep fitness unchanged
        return entity
