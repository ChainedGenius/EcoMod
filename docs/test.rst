function package
================

Submodules
----------

function.main module
--------------------


TEST LATEX
__________
 :math:`\overline{s} = g\left(\overline{m}, \overline{s}, \overline{x}, \overline{m} \right)`

 :math:`0 = \left[ f\left(\overline{m}, \overline{s}, \overline{x}, \overline{m}, \overline{s}, \overline{x} \right) \right]`

TESt CODE
___________
.. code:: python

    N = 10000

    vec_m = m[None,:].repeat(N, axis=0) # we repeat each line N times
    vec_s = s[None,:].repeat(N, axis=0) # we repeat each line N times
    vec_x = x[None,:].repeat(N, axis=0)
    vec_X = X[None,:].repeat(N, axis=0)
    vec_p = p[None,:].repeat(N, axis=0)
    # actually, except for vec_s, the function repeat is not need since broadcast rules apply
    vec_s[:,0] = linspace(2,4,N) # we provide various guesses for the steady-state capital
    vec_S = vec_s



New part_new string mines
___________________

New part_new string plus
+++++++++++