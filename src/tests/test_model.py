import pytest
import traffic_model as model


@pytest.mark.parametrize(
    ('lane', 'position', 'result'),
    [
        (1, 5, 9),
        (1, 4, 6),
        (1, 10, 3),
    ]
)
def test_first_station(lane, position, result):
    stations = (3, 6, 9)
    model.Bus.set_station(stations)
    bus = model.Bus(lane, position)
    assert bus.get_station == result


"""
def test_first_station():
    stations = (3, 6, 9)
    model.Bus.set_station(stations)
    bus = model.Bus(1, 5)
    assert bus.get_station == 9
"""