from firedrake import *
import ufl
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)

u.interpolate(Constant(1.0))

J = inner(grad(u), grad(u)) * dx
# J = u * dx
# J = grad(u)[0] * dx
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
form = ufl.replace(form, {SpatialCoordinate(mesh): ufl.classes.ReferenceValue(mesh.coordinates)})
print("form %s" % (form))
w = TestFunction(mesh.coordinates.function_space())
d = derivative(form, mesh.coordinates, ufl.classes.ReferenceValue(w))
shape_derivative = apply_derivatives(d)
print("-----------")
print("Shape derivative: ", shape_derivative)
print("State: ", u)
print("Coordinate: ",  mesh.coordinates)
print("Direction: ", w)
assemble(shape_derivative)
