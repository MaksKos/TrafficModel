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
NUM_POINTS = 100
CORES = 6
TIME_STOP = 20

# main settings
v_max = 3
# p_cl = 0.8 default
# p_slow = 0.5 default
t_s = 5_000
t_e = 1000
veh_lenght = 2
n_cells = 1000
n_lane = 1
# save data folder
directory = 'data/'
file_name = 'fundamental_bus'

# variable 
steps = [None, 1000, 250, 100, 50, 10]
#############################################################

if not os.path.isdir(directory):
    os.makedirs(directory)

def calc_data(step_stations: int, cluster: Client):

    
    road_param = {
        "N_cells": n_cells,
        "N_lane": n_lane,
    }

    if step_stations is None:
        stations = 0
    else:
        stations = tuple([i for i in range(0, n_cells, step_stations)])
    
    fun_step = setting_step(road_param, t_s, t_e, v_max, stations)
    buses_arr = np.linspace(0, n_cells*n_lane//veh_lenght, NUM_POINTS+1)[1:].astype(int)
    futs = []
    for buses in buses_arr:
        work = cluster.submit(fun_step, buses, pure=False)
        futs.append(work)
    
    progress(futs,)
    print("\n")
    results = cluster.gather(futs)
    return results

def setting_step(road_set, t_s, t_e, v_max, stations):
    def step(buses):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        model.Bus._vel_max = v_max
        model.Bus.set_station(stations)
        model.Bus._stop_step = TIME_STOP
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

    for step_station in steps:
        print(f"Calculate model step: {step_station}")
        result = calc_data(step_station, client)
        tabel = pd.DataFrame(result)
        name = directory + file_name + f"_step_{step_station}_v_{v_max}.csv"
        tabel.to_csv(name)
    
    client.close()
    return 0

if __name__ == "__main__":
    
    main()