from firedrake import *
import ufl
mesh = UnitSquareMesh(3, 3)
# mesh = UnitIntervalMesh(4)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
(x, y) = SpatialCoordinate(mesh)
u.interpolate(x)

w = TestFunction(mesh.coordinates.function_space())

# J = inner(grad(u), grad(u)) * dx
# deriv = -2*inner(dot(grad(w), grad(u)), grad(u)) * dx + inner(grad(u), grad(u)) * div(w) * dx
# J = u * dx
# deriv = div(w) * u * dx
f = x*x + y*y*x
J = f * dx
deriv = div(f*w) * dx
from ufl.algorithms.compute_form_data import *
form = J
form = apply_algebra_lowering(form)
form = apply_derivatives(form)
form = group_form_integrals(form, J.ufl_domains())
form = apply_function_pullbacks(form)
form = apply_integral_scaling(form)
form = apply_default_restrictions(form)
form = apply_geometry_lowering(form)
form = apply_derivatives(form)
form = apply_geometry_lowering(form)
form = apply_derivatives(form)
print("form before replacing %s" % (form))
form = ufl.replace(form, {SpatialCoordinate(mesh): ufl.classes.ReferenceValue(mesh.coordinates)})
print("form after replacing %s" % (form))
d = derivative(form, mesh.coordinates, ufl.classes.ReferenceValue(w))
shape_derivative = apply_derivatives(d)
print("-----------")
print("Shape derivative: ", shape_derivative)
print("State: ", u)
print("Coordinate: ",  mesh.coordinates)
print("Direction: ", w)
shape_derivative._cache['shape'] = True
computed = assemble(shape_derivative).vector()[:]
actual = assemble(deriv).vector()[:]
import numpy as np
print("Error:", np.linalg.norm(computed-actual)/np.linalg.norm(actual))
