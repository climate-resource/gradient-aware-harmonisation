import pint
import pint.testing

from openscm_units import unit_registry

from gradient_aware_harmonisation.harmonise import harmonise


Q = unit_registry.Quantity

def test_basic_harmonisation():
    # Switch to using continuous timeseries here, for this first test.
    # Then we can deal with array inputs in a later step.
    x_target = Q([2010, 2015, 2020], "yr")
    y_target = Q([1.0, 2.0, 3.0], "ppm")

    x_harmonisee = Q([2020, 2030, 2050], "yr")
    y_harmonisee = Q([3.5, 6.5, 10.0], "ppm")



    res = harmonise(
        f1=(x_target, y_target),
        f2=(x_harmonisee, y_harmonisee),
        x0=Q(2020, "yr")
        #convergence_year=Q(2050, "yr")
    )

    # Testing harmonisation
    # Absolute values
    # (you could do this better if you checked the time points
    # more carefully rather than hard-coding the indexes)
    pint.testing.assert_equal(res["f1"][1][res["x0"]], res["f2"][1][x0])
    # Gradients (need to switch to gradients below)
    #pint.testing.assert_equal(y_res[0], y_target[-1])

    # Test the convergence
    # Absolute values
    # (Need to fix the indexing below)
    pint.testing.assert_equal(y_res[2050], y_harmonisee[2050])
    # Gradients (need to switch to gradients below)
    pint.testing.assert_equal(y_res[2050], y_harmonisee[2050])