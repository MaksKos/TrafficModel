import pytest
import numpy as np
import traffic_model.traffic_model as model


@pytest.mark.parametrize(
    ('lane', 'position', 'result'),
    [
        (1, 5, 2),
        (1, 4, 1),
        (1, 10, 0),
    ]
)
def test_first_station(lane, position, result):
    stations = (3, 6, 9)
    model.Bus.set_station(stations)
    bus = model.Bus(lane, position)
    assert bus.station == result

def test_bus_initial_position():
    empty_cells = np.array([[0, 0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6]])
    amount = 3
    position, cells = model.Bus.initial_position(empty_cells=empty_cells, amount=amount, shape=())
    assert position.shape == (amount, 2)
    assert cells.shape == (len(empty_cells)-amount*model.Bus._lenght, 2)

def test_bus_move():
    assert True


# make the same tests for <HumanDriveVehicle>
def test_road_deque():
    stations = (3, 6, 9)
    road_param = {
        "N_cells": 100,
        "N_lane": 1,
    }
    cars = {model.Bus: 5, model.HumanDriveVehicle: 10}
    sum_cars = 15
    model.Bus.set_station(stations)
    road = model.Model( road_param, cars)
    road_deque = road.road_model
    count = 0
    for lane in road_deque:
        if not lane:
            continue
        count += len(lane)
        for index in range(len(lane)-1):
            assert lane[index].position < lane[index+1].position
    assert count == sum_cars


def test_cell_overflow():
    road_param = {
        "N_cells": 100,
        "N_lane": 2,
    }
    buses = {model.HumanDriveVehicle: 60}
    station = tuple([50, 100, 200, 400, 600, 900])
    model.Bus.set_station(station)
    model_bus = model.Model(road_parametrs=road_param, vehicles=buses)
    model_bus.model_research(400, is_diagramm=True)
    occupate_cells = sum([num*typ._lenght for typ, num in buses.items()])
    diagramm = model_bus.get_data()
    occupate_check = np.array([[np.sum(layer != None) for layer in lane] for lane in diagramm.values()])
    occupate_check = np.sum(occupate_check, axis=0)
    assert len(np.unique(occupate_check)) == 1
    assert occupate_check[0] == occupate_cells
