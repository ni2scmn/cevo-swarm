from pathlib import Path
import sys

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

        if random_seed is None:
            self.random_seed = random.randint(0, 100000000)
        else:
            self.random_seed = random_seed

        np.random.seed(int(self.random_seed))

        self.swarm = self.init_swarm(self.cfg)

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
        self.export_data = self.cfg.get("export_data")
        self.export_steps = self.cfg.get("export_steps")


        self.population_size = self.cfg.get("train", "population_size")
        self.n_generations = self.cfg.get("train", "n_generations")
        self.metric = self.cfg.get("train", "metric")
        self.mutation_rate = self.cfg.get("train", "mutation_rate")
        self.crossover_rate = self.cfg.get("train", "crossover_rate")
        self.elitism = self.cfg.get("train", "elitism")

        self.population = self.init_population()
        self.set_population()

        # self._init_log()


    def run(self):
        if self.verbose:
            print("Running with seed: %d" % self.random_seed)
    
        
        # Run the simulation
        self.run_episode()

        # Save the results
        if self.export_data:
            # self.log_CA_data()
            # self.log_data()
            pass

        if self.verbose:
            print("\nSimulation completed.")

    def run_episode(self):
        if self.verbose:
            print("Running with seed: %d" % self.random_seed)
        while self.warehouse.counter <= self.cfg.get("time_limit"):
            self.iterate()
            # if self.export_data:
            #     self.log_CA_data()
            #     if self.warehouse.counter in self.export_ts:
            #         self.log_data()

        if self.verbose:
            print("\n")

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            control_network = FeedforwardNN(
                layers=self.cfg.get("robot", "nn_layers"),
                weight_init=random_weight_init,
                activation_fun=self.cfg.get("robot", "activation_funcs"),
            )
            population.append(control_network)
        return population
    
    def set_population(self):
        for i in range

    def init_swarm(self, cfg):
        swarm = Swarm(
            repulsion_o=cfg.get("warehouse", "repulsion_object"),
            repulsion_w=cfg.get("warehouse", "repulsion_wall"),
            heading_change_rate=cfg.get("heading_change_rate"),
        )

        nn_layers = cfg.get("robot", "nn_layers")
        weight_init = cfg.get("robot", "weight_init")
        if weight_init == "random":
            weight_init_fun = random_weight_init
        else:
            raise ValueError("Unknown weight init function")
        activation = cfg.get("robot", "activation_funcs")
        activation_funcs = []

        for af in activation:
            if af == "sigmoid":
                activation_funcs.append(sigmoid)
            elif af == "softmax":
                activation_funcs.append(softmax)
            else:
                raise ValueError("Unknown activation function")

        # control_network = FeedforwardNN(
        #     layers=nn_layers,
        #     weight_init=weight_init_fun,
        #     activation_fun=activation_funcs,
        # )
        # belief_space = NNBeliefSpace(
        #     bs_nn_weights=np.random.uniform(-1, 1, size=control_network.get_weights().shape)
        # )

        for _ in range(cfg.get("warehouse", "number_of_agents")):
            control_network = FeedforwardNN(
                layers=nn_layers,
                weight_init=weight_init_fun,
                activation_fun=activation_funcs,
            )
            belief_space = NNBeliefSpace(
                bs_nn_weights=np.random.uniform(-1, 1, size=control_network.get_weights().shape)
            )
            robot_obj = Robot(
                cfg.get("robot", "radius"),
                cfg.get("robot", "max_v"),
                camera_sensor_range=cfg.get("robot", "camera_sensor_range"),
                control_network=control_network,
                belief_space=belief_space,
            )
            swarm.add_agents(robot_obj, 1)

        swarm.generate()
        swarm.init_params(cfg)
        return swarm


    def iterate(self):
        self.warehouse.iterate(self.cfg.get("heading_bias"), self.cfg.get("box_attraction"))
        counter = self.warehouse.counter

        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter % 100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(counter)

    def evaluate_fitness(self):
        
    
    def exit_sim(self, counter):
        if counter > self.cfg.get("time_limit"):
            print("Exiting...")
            self.exit_threads = True