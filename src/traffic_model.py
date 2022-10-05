import numpy as np
from collections import deque

class Bus():

    _bus_lenght = 2
    _vel_max = 3

    def __init__(self, line: int, position: int) -> None:
        pass

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):

        bus_line = int(0)
        buses_cells = empty_cells[empty_cells.T[0] == bus_line]
        buses_cells = buses_cells[::Bus._bus_lenght]
        if buses_cells.shape[0] < amount:
            raise ValueError("No free cells for buses")
        index_position = np.sort(np.random.choice(buses_cells.T[1], size=amount, replace=False))
        buses_position = buses_cells[index_position]
        for i in range(Bus._bus_lenght):
            empty_cells = np.delete(empty_cells, index_position+i, axis=0)
        return buses_position, empty_cells

class HumanDriveVehicle():

    _vel_max = 3

    def __init__(self, line: int, position: int) -> None:
        pass

    @staticmethod
    def initial_position(empty_cells, shape: tuple, amount: int):

        index_position = np.sort(np.random.choice(empty_cells.shape[0], size=amount, replace=False))
        hdv_position = empty_cells[index_position]
        empty_cells = np.delete(empty_cells, index_position, axis=0)
        return hdv_position, empty_cells

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
        empty_cells = np.indices((self.n_line, self.n_cells), dtype=np.int32).transpose(1, 2, 0).reshape(self.n_cells*self.n_line, 2)
        vehicle_position = {}
        # position initialization of each vehicle type
        shape = (self.n_line, self.n_cells)
        for vehicle_type, amount in vehicles.items():
            vehicle_position[vehicle_type], empty_cells = vehicle_type.initial_position(empty_cells, shape, amount)
        self.road_model = self.__make_road_deque(vehicle_position)


    def __make_road_deque(self, veh_pos: dict):

        road_matrix = [[None for j in range(self.n_cells)] for _ in range(self.n_line)]
        for veh_type, positions in veh_pos.items():
            for position in positions:
                line = position[0]
                cell = position[1]
                road_matrix[line][cell] = veh_type(line, cell)

        road_model = []
        for line in road_matrix:
            road_model.append(deque(set(line).remove(None)))
        return road_model

    def step(self):

        self.__get_distance()
        if self.n_line > 1:
            for index, line in enumerate(self.road_model):
                if index == 0:
                    pass
                elif index == self.n_line-1:
                    pass
                else:
                    pass
        self.__get_distance()  
        #vehicle move

    def __get_distance(self):
        pass
     
    def __line_change(self):
        pass

    def model_stabilization(self, n_step):
        pass

    def model_research(self, n_step):
        pass

    def get_data(self):
        pass

        




        
        


