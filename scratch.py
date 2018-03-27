from firedrake import *
import numpy as np
import ufl
mesh = UnitSquareMesh(3, 3)
# mesh = UnitIntervalMesh(4)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
v = TestFunction(V)
(x, y) = SpatialCoordinate(mesh)
u.interpolate(x)

w = TestFunction(mesh.coordinates.function_space())

def run_test(J, deriv, shape_deriv):
    if deriv is not None:
        computed = assemble(derivative(J, u)).vector().get_local()
        actual = assemble(deriv).vector().get_local()
        diff = np.linalg.norm(computed-actual)
    else:
        diff = None

    computed = assemble(derivative(J, mesh.coordinates)).vector().get_local()
    actual = assemble(shape_deriv).vector().get_local()
    shape_diff = np.linalg.norm(computed-actual)
    print("Diff: ", diff)
    print("Shape Diff: ", shape_diff)

J = inner(grad(u), grad(u)) * dx
deriv = 2 * inner(grad(u), grad(v)) * dx
shape_deriv = -2*inner(dot(grad(w), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(w) * dx
run_test(J, deriv, shape_deriv)

J = u * u * dx
deriv = 2 * u * v * dx
shape_deriv = u * u * div(w) * dx
run_test(J, deriv, shape_deriv)

f = x * y
J = f * dx
shape_deriv = div(f*w) * dx
run_test(J, None, shape_deriv)

J = sin(inner(u, u)) * dx
deriv = cos(inner(u, u)) * 2 * u * v * dx
shape_deriv = sin(inner(u, u)) * div(w) * dx


run_test(J, deriv, shape_deriv)

import sys; sys.exit(0)

assemble(derivative(J, mesh.coordinates, ufl.classes.ReferenceValue(w)))
# J = u * dx
# deriv = div(w) * u * dx
# f = x*x + y*y*x
# J = f * dx
# deriv = div(f*w) * dx
from ufl.algorithms.compute_form_data import *
form = J
form = apply_algebra_lowering(form)
form = apply_derivatives(form)
form = group_form_integrals(form, J.ufl_domains())
form = apply_function_pullbacks(form)
form = apply_integral_scaling(form)
form = apply_default_restrictions(form)
print("1, ", form)
form = apply_geometry_lowering(form)
print("2, ", form)
form = apply_derivatives(form)
print("3, ", form)
form = apply_geometry_lowering(form)
print("4, ", form)
form = apply_derivatives(form)
print("5, ", form)
print("form before replacing %s" % (form))
form = ufl.replace(form, {SpatialCoordinate(mesh): ufl.classes.ReferenceValue(mesh.coordinates)})
print("form after replacing %s" % (form))
# d = derivative(form, mesh.coordinates, ufl.classes.ReferenceValue(w))
d = derivative(form, SpatialCoordinate(mesh), ufl.classes.ReferenceValue(w))
shape_derivative = apply_derivatives(d)
print("-----------")
print("Shape derivative: ", shape_derivative)
print("State: ", u)
print("Coordinate: ",  mesh.coordinates)
print("Direction: ", w)
shape_derivative._cache['shape'] = True
computed = assemble(shape_derivative).vector()[:]
actual = assemble(deriv).vector()[:]
print("Error:", np.linalg.norm(computed-actual)/np.linalg.norm(actual))
