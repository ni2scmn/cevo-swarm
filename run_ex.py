from simulator.wh_sim import *
from simulator.lib import Config, SaveTo
from simulator import CFG_FILES
import time
import numpy as np

###### Functions ######

def run_experiment(cfg_obj, st, ex_id,  r_i):
    seed = np.random.randint(0, 10000000)
    sim = Simulator(cfg_obj, verbose=False, random_seed=seed)
    
    sim.run()
    if True:
        # for key in [
        #     "P_m",
        #     "D_m",
        #     "SC",
        #     "r0",
        #     "BS_P_m",
        #     "BS_D_m",
        #     "BS_SC",
        #     "BS_r0",
        #     "r_phase",
        #     "influence_rates",
        #     "resistance_rates",
        # ]:
        #     data = sim.CA_data[key]
        #     st.export_data(ex_id, data, key + "_" + str(r_i), transpose=True)

        # @TODO remove self_updates variable ? -- data already logged in r_phase
        # for key in ["social_transmission", "self_updates"]:
        #     data = sim.CA_data[key]
        #     records = [{"timestep": i, key: v} for i, v in data.items()]
        #     st.export_data(ex_id, records, key+ "_" + str(r_i))

        dn = st.export_data(ex_id, sim.data["ap_distance"].values(), "ap_distance"+ "_" + str(r_i))
        st.export_data(ex_id, sim.data["x_axis"].values(), "x_axis"+ "_" + str(r_i))
        st.export_data(ex_id, sim.data["y_axis"].values(), "y_axis"+ "_" + str(r_i))

        st.export_data(ex_id, sim.data["nn_weights"], "nn_weights"+ "_" + str(r_i), transpose=True)
        st.export_data(ex_id, sim.data["belief_space_weights"], "belief_space_weights"+ "_" + str(r_i))

        #dn = st.export_data(ex_id, sim.data["box_c"], "boxes"+ "_" + str(r_i))
        #st.export_data(ex_id, sim.data["rob_c"], "robots"+ "_" + str(r_i))
        st.export_metadata(
            dn, {"box_type_ratio": cfg_obj.get("box_type_ratio"), "ap": cfg_obj.get("ap")}
        )




###### Run experiment ######

if __name__ == "__main__":

    ###### Experiment parameters ######

    ex_id = "e_nn_1"
    verbose = False
    export_data = True

    ###### Config class ######

    default_cfg_file = CFG_FILES["default"]
    cfg_file = CFG_FILES["ex_nn_1"]
    cfg_obj = Config(cfg_file, default_cfg_file, ex_id=ex_id)

    t0 = time.time()
    st = SaveTo()

    runs = int(cfg_obj.get("runs"))
    
    print(ex_id + "/" + str(st.ts))

    # Use multithreading to run multiple experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(run_experiment, [cfg_obj] * runs, [st] * runs, [ex_id] * runs, range(runs)),
                total=runs,
                desc="Running experiments",
            )
        )

    # non parallel version
    # for r_i in range(runs):
    #     run_experiment(cfg_obj, st, ex_id, r_i)

        

    t1 = time.time()
    dt = t1 - t0
    print("Time taken: %s" % str(dt), "\n")
