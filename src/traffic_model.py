from turtle import position
import numpy as np


class Bus():


    def __init__(self) -> None:
        pass

    @staticmethod
    def initial_position(empty_cells, shape, count):

        bus_line = int(0)
        n_cells = empty_cells[0:shape[1]][1]
        position = np.sort(np.random.choice(n_cells, size=count, replace=False))

        bus_
        return bus_position


class HumanDriveVehicle():

    def __init__(self) -> None:
        pass

    
    @staticmethod
    def initial_position(empty_cells, shape, count):
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
        empty_cells = list(np.indices((self.n_line, self.n_cells)).transpose(1, 2, 0).reshape(self.n_cells*self.n_line, 2))
        self.vehicle_position = {}
        # position initialization of each vehicle type
        shape = (self.n_line, self.n_cells)
        for vehicle_type, count in vehicles.items():
            self.vehicle_position[vehicle_type] = vehicle_type.initial_position(empty_cells, shape, count)
        self.road_model = self.__make_road_deque()


    def __make_road_deque(self):
        road_matrix = [[j for j in range(self.n_cells)] for _ in range(self.n_line)]
        pass

    def step(self):
        pass

    def model_stabilization(self, n_step):
        pass

    def model_research(self, n_step):
        pass

    def get_data(self):
        pass

        




        
        


