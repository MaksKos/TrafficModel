import os
import sys

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, progress

modul_path = os.path.abspath(r"..\src\traffic_model")
if not modul_path in sys.path:
    sys.path.append(modul_path)

import traffic_model as model


#############################################################
STEP = 1
CORES = 6
# main settings
v_max = 3
# p_cl = 0.8 default
# p_slow = 0.5 default
t_s = 10_000
t_e = 2000
n_lane = 1

# save data folder
directory = 'data/'
file_name = 'bus'
# variable 
steps = [1000, 1000, 200, 100, 50, 50, 50, 100, 100]
n_station = [0, 1, 5, 10, 20, 5, 10, 5, 20]
#############################################################

if not os.path.isdir(directory):
    os.makedirs(directory)

def calc_data(step_stations: int, amount_station:int, cluster: Client):

    if amount_station < 1:
        n_cells = step_stations
    else:
        n_cells = step_stations*amount_station
    
    road_param = {
        "N_cells": n_cells,
        "N_lane": n_lane,
    }
    if amount_station < 1:
        stations = 0
    else:
        stations = tuple([i for i in range(0, n_cells, step_stations)])

    fun_step = setting_step(road_param, t_s, t_e, v_max, stations)
    buses_arr = np.arange(1, 101)
    futs = []
    for buses in buses_arr:
        work = cluster.submit(fun_step, buses, pure=False)
        futs.append(work)
    
    progress(futs)
    print("\n")
    results = cluster.gather(futs)
    return results

def setting_step(road_set, t_s, t_e, v_max, station):
    def step(buses):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        model.Bus._vel_max = v_max
        model.Bus.set_station(station)
        buses_type = {model.Bus: buses}
        road = model.Model(road_parametrs=road_set, vehicles=buses_type)
        road.model_stabilization(t_s)
        road.model_research(t_e)
        return road.result
    return step

def main():

    cluster= LocalCluster(n_workers = CORES, threads_per_worker = 1)
    client = Client(cluster)
    print(client)

    for step_station, amount_stations in zip(steps, n_station):
        print(f"Calculate model step: {step_station}; amount station: {amount_stations}")
        result = calc_data(step_station, amount_stations, client)
        tabel = pd.DataFrame(result)
        name = directory + file_name + f"_step_{step_station}_station_{amount_stations}.csv"
        tabel.to_csv(name)
    
    client.close()
    return 0

if __name__ == "__main__":
    
    main()

