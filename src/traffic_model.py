from re import A
import numpy as np


class Bus():

    def __init__(self) -> None:
        pass

    @staticmethod
    def initial_position(road_matrix):
        bus_position = None
        return bus_position


class HumanDriveVehicle():

    def __init__(self) -> None:
        pass

    
    @staticmethod
    def initial_position(road_matrix):
        car_position = None
        return car_position

class Model():

    __vehicle_type = [Bus, HumanDriveVehicle]

    def __init__(self, road_parametrs: dict, vehicles: dict) -> None:
        """
        Initialization of traffic model

        Args:
            road_parametrs (dict): N_line - numbers of line, N_cells - lenght of road 
            vehicles (dict): <vehicle_type <class>: amount (int), ...> - dict of vehicle types and amount 
        """
        for vehicle_type in vehicles:
            if not (vehicle_type in self.__vehicle_type):
                raise TypeError(f"Unknown vehicle type ({vehicle_type})")
        
        self.n_line = road_parametrs['N_line']
        self.n_cells = road_parametrs['N_cells']
        self.road_matrix = [[j for j in range(self.n_cells)] for _ in range(self.n_line)]
        empty_cells = list(np.indices((self.n_line, self.n_cells)).transpose(1, 2, 0).reshape(self.n_cells*self.n_line, 2))
        self.vehicle_position = {}
        # position initialization of each vehicle type
        for vehicle_type in vehicles:
            self.vehicle_position[vehicle_type] = vehicle_type.initial_position(empty_cells)


        




        
        


