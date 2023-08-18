import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
#from dask.distributed import Client, LocalCluster, progress

modul_path = os.path.abspath(r"..\src\traffic_model")
if not modul_path in sys.path:
    sys.path.append(modul_path)

import traffic_model as model

#print("Calculate with statio_step - 35")
print("Calculate with capacity - 40")
#############################################################
POINTS = 10 # кол-во рассматриваемых случае - разбтение пропорции [0, 1]
CORES = 4
# main settings
v_max = 3
# p_cl = 0.8 default
# p_slow = 0.5 default
t_s = 4000 #10_000
t_e = 4000 #2000
n_lane = 2
n_cells = 1000
bus_cap = 40 #80
station_step = 70 #70

# save data folder
directory = 'data/'
file_name = 'table_cap'

# variable 
peoples_list = [100] # [100, 300, 1000, 1600]
#############################################################

if not os.path.isdir(directory):
    os.makedirs(directory)

# pre - calculation
proportion = np.linspace(0,1, POINTS+1)
station_list = tuple([i for i in range(0, n_cells, station_step)])
road_param = {
        "N_cells": n_cells,
        "N_lane": n_lane,
    }

def main():

    #cluster= LocalCluster(n_workers = CORES, threads_per_worker = 1)
    #client = Client(cluster)
    #print(client)

    for n_people in peoples_list:
        print(f"Calculate model with {n_people} peoples")
        result = calc_data(n_people) # , client)

        tabel_city = pd.DataFrame(result).drop(labels=['velosity_av_typed'], axis=1)
        tab_veh = [layer['velosity_av_typed'] for layer in result]
        tabel_vehicle = pd.DataFrame(tab_veh)

        name = directory + file_name
        tabel_city.to_csv(name + f"_city_{n_people}.csv")
        tabel_vehicle.to_csv(name + f"_veh_{n_people}.csv")

    #client.close()
    return 0

def calc_data(peoples: int):#, cluster: Client):
 
    fun_step = setting_step(road_param, t_s, t_e, v_max, station_list)

    passenger = peoples*proportion
    drivers = peoples - passenger.astype(int)
    buses = np.ceil(passenger/bus_cap).astype(int)

    paralell_pull = Parallel(CORES, verbose=100)
    futs = []
    for buses_i, drivers_i in zip(buses, drivers):
        futs.append(delayed(fun_step)(buses_i, drivers_i))
    
    #progress(futs)
    print("\n")
    results = paralell_pull(futs)
    return results

def setting_step(road_set: dict, t_s: int, t_e: int, v_max: int, stations: tuple):
    def step(buses: int, drivers: int):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        model.Bus.set_station(stations)
        model.Bus._vel_max = v_max
        cars_type = {model.Bus: int(buses), model.HumanDriveVehicle: int(drivers)}
        road = model.Model(road_parametrs=road_set, vehicles=cars_type)
        road.model_stabilization(t_s)
        road.model_research(t_e)
        return road.result
    return step

if __name__ == "__main__":
    
    main()
