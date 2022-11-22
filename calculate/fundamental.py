import os
import timeit
import sys

import numpy as np
import dask
import pandas as pd
from dask.distributed import Client, progress

modul_path = os.path.abspath(r"..\src\traffic_model")
if not modul_path in sys.path:
    sys.path.append(modul_path)

import traffic_model as model


#############################################################
STEP = 50

# main settings
v_max = 5
# p_cl = 0.8 default
# p_slow = 0.5 default
t_s = 5_000
t_e = 500

n_cells = 1000
# save data folder
directory = 'data/'
file_name = 'fundamental'
# variable 
n_lane = [1, 2]

#############################################################

if not os.path.isdir(directory):
    os.makedirs(directory)

def calc_data(n_lane):

    client = Client()
    print( "Workers:" + str(len( list(client.scheduler_info()['workers']) )) )

    road_param = {
        "N_cells": n_cells,
        "N_lane": n_lane,
    }
    fun_step = setting_step(road_param, t_s, t_e, v_max)
    cars_arr = np.arange(n_cells, step=STEP)[1:5]
    
    print(f"Calculate N lane: {n_lane}")
    results = []
    for cars in cars_arr:
        fun = dask.delayed(fun_step)(cars)
        results.append(fun)

    results = dask.persist(*results)
    progress(results)
    results.compute()

    client.close()
    return results

def setting_step(road_set, t_s, t_e, v_max):
    def step(cars):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        model.HumanDriveVehicle._vel_max = v_max
        cars = {model.HumanDriveVehicle: cars}
        road = model.Model(road_parametrs=road_set, vehicles=cars)
        road.model_stabilization(t_s)
        road.model_research(t_e)
        return road.result
    return step

def main():

    for n in n_lane:
        result = calc_data(n)
        tabel = pd.DataFrame(result)
        name = directory + file_name + f"_lane_{n}_v_{v_max}.csv"
        tabel.to_csv(name)

if __name__ == "__main__":
    
    main()

