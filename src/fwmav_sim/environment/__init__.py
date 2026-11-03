"""Environment models (M1 Task 6).

Relative airflow at each element is v_rel = v_element - v_air (brief §5.4);
there is no inflow model beyond relative velocity (assumption A-006).
"""

from fwmav_sim.environment.spec import environment_from_spec
from fwmav_sim.environment.still_air import still_air
from fwmav_sim.environment.uniform_wind import uniform_wind

__all__ = ["environment_from_spec", "still_air", "uniform_wind"]
