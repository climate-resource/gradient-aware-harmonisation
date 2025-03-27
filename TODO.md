- Figure out how hard it would be to get this product of splines to support derivative and antiderivative
- Figure out how to use autodiff for our derivatives and antiderivatives

- Add any other tests of harmonise_splines we think are useful

- Think about which other parts of the API we still need, e.g.
    - can we remove `interpolate_harmoniser` (check)
    - we can definitely remove `ConvergenceMethod` (check)

- Add a `to_spline` method to the `Timeseries` class (including tests)
    - this can be very thin (it's basically syntactic sugar),
      all it needs to do is call `timeseries_to_spline` really (check)

- fix up the tests in `test_harmonisation_integration`
- any other general clean up
    - e.g. moving functions into their own modules

- docs page on harmonisation with splines directly (start from `scratch.py`)

Hoped result:

- strip our API down to as few functions as possible
    - identify exactly what we need for harmonisation
      so it's clearer what is low-level and what is higher-level
- code coverage during testing above 90%
