from util import np, _, freeze
import pygmsh
import meshio
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation as pltTriangulation
from typing import Callable

# A class' member function tagged as cached property
# will remember whatever is returned the first time it's called.
# The second time it's called it will not be computed again.
# We have to make sure that whatever is returned is immutable though, meaning
# that it cannot be changed from outside of the class.
# If an array is returned, we can freeze that array for example (see below).
from functools import cached_property


class Triangulation:

  @staticmethod
  def from_polygon(*args, **kwargs):
    return mesh_from_polygon(*args, **kwargs)

  @classmethod
  def from_file(cls, filename):
    """ Load mesh from gmsh file. """
    from meshio import gmsh
    return cls(gmsh.main.read(filename))

  mesh: meshio._mesh.Mesh

  def __init__(self, mesh):
    assert isinstance(mesh, meshio._mesh.Mesh)

    simplex_names = ('line', 'triangle', 'vertex')
    if not set(mesh.cells_dict.keys()).issubset(set(simplex_names)):
      raise NotImplementedError("Expected the mesh to only contain the simplices:"
                                " '{}' but found '{}'."
                                .format(simplex_names, tuple(mesh.cells_dict.keys())))

    self.mesh = mesh

  @property
  @freeze
  def triangles(self):
    return self.mesh.cells_dict['triangle']

  @property
  @freeze
  def lines(self):
    """ Return array ``x`` of shape (nboundaryelements, 2) where x[i] contains
        the indices of the vertices that fence-off the i-th boundary element. """
    return self.mesh.cells_dict['line']

  @cached_property
  @freeze
  def normals(self):
    # get all forward tangents
    ts = (self.points[self.lines] * np.array([-1, 1])[_, :, _]).sum(1)

    # normal is tangent[::-1] * [1, -1]
    ns = ts[:, ::-1] * np.array([[1, -1]])

    # normalise
    return ns / np.linalg.norm(ns, ord=2, axis=1)[:, _]

  @property
  @freeze
  def points(self):
    # mesh.points.shape == (npoints, 3) by default, with zeros on the last axis.
    # => ignore that part in R^2.
    return self.mesh.points[:, :2]

  def points_iter(self):
    """
      An iterator that returns the three vertices of each element.

      Example
      -------

      for (a, b, c) in mesh.points_iter():
        # do stuff with vertices a, b and c

    """
    for tri in self.triangles:
      yield self.points[tri]

  def plot(self):
    plot_mesh(self)

  @cached_property
  @freeze
  def BK(self):
    """
      Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BK[i, :, :] or, in short, mesh.BK[i] gives
      the Jacobi matrix corresponding to the i-th element.

      Example
      -------

      for i, BK in enumerate(mesh.BK):
        # do stuff with the Jacobi matrix BK corresponding to the i-th element.
    """
    a, b, c = self.points[self.triangles.T]
    # freeze the array to avoid accidentally overwriting stuff
    return np.stack([b - a, c - a], axis=2)

  @cached_property
  @freeze
  def detBK(self):
    """ Jacobian determinant (measure) per element. """
    # the np.linalg.det function returns of an array ``x`` of shape
    # (n, m, m) the determinant taken along the last two axes, i.e.,
    # in this case an array of shape (nelems,) where the i-th entry is the
    # determinant of self.BK[i]
    return np.abs(np.linalg.det(self.BK))

  @cached_property
  @freeze
  def detBK_boundary(self):
    # XXX: docstring
    a, b = self.points[self.lines.T]
    return np.linalg.norm(b - a, ord=2, axis=1)

  @cached_property
  @freeze
  def BKinv(self):
    """
      Inverse of the Jacobi matrix per element of shape (nelems, 2, 2).
      mesh.BKinv[i, :, :] or, in short, mesh.BKinv[i] gives
      the nverse Jacobi matrix corresponding to the i-th element of shape (2, 2).
    """
    (a, b), (c, d) = self.BK.T
    return np.rollaxis(np.stack([[d, -b], [-c, a]], axis=1), -1) / self.detBK[:, _, _]

  @cached_property
  @freeze
  def boundary_indices(self):
    """ Return the sorted indices of all vertices that lie on the boundary. """
    return np.sort(np.unique(self.lines.ravel()))

  def tripcolor(self, z, title=None):
    """ Plot discrete data ``z`` on the vertices of the mesh.
        Data is linearly interpolated between the vertices. """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    triang = pltTriangulation(*self.points.T, self.triangles)
    tpc = ax.tripcolor(triang, z, shading='flat', edgecolor='k')
    fig.colorbar(tpc)
    if title is not None:
      ax.set_title(title)
    plt.show()


def mesh_from_polygon(points: np.ndarray, mesh_size=0.05) -> Triangulation:
  """
    create :class: ``Triangulation`` mesh from ordered set of boundary
    points.

    parameters
    ----------
    points: Array-like of shape points.shape == (npoints, 2) of boundary
            points ordered in counter-clockwise direction.
            The first point need not be repeated.
    mesh_size: Numeric value determining the density of cells.
               Smaller values => denser mesh.
  """

  points = np.asarray(points)
  assert points.shape[1:] == (2,)

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(points, mesh_size=mesh_size)
    mesh = geom.generate_mesh(algorithm=5)

  return Triangulation(mesh)


def plot_mesh(mesh: Triangulation):
  """ Plot a mesh of type ``Triangulation``. """

  points = mesh.points
  triangles = mesh.triangles

  fig, ax = plt.subplots()

  ax.set_aspect('equal')

  ax.triplot(*points.T, triangles)
  plt.show()


def plot_function_on_mesh(mesh: Triangulation, func: Callable):
  """
    Plot scalar function ``func`` on ``mesh``.

    parameters
    ----------
    mesh: :class: ``Triangulation`` generated by, for instance ``mesh_from_polygon``.
    func: Callable func = func(x, y) wherein x and y are the mesh vertices.
          Needs to be vectorized, i.e, z = f(x, y) with x.shape == y.shape == (N,)
          means z.shape == (N,).
  """

  points = mesh.points

  # triangles representing the mesh
  # triangles is an array of shape (npoints, 3), where triangles[i] = [i0, i1, i2]
  # means that the i-th cell's vertices are, in counter-clockwise direction,
  # given by points[ [i0, i1, i2] ].
  triangles = mesh.triangles

  # create the function value to be put on the z-axis
  z = func(*points.T)

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.plot_trisurf(*points.T, z, triangles=triangles,
                                linewidth=0.5,
                                antialiased=True,
                                alpha=.8,
                                cmap=plt.cm.autumn, edgecolors='k')

  plt.show()


if __name__ == '__main__':
  xi = np.linspace(0, 2*np.pi, 41)[:-1]
  circle = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  mesh = mesh_from_polygon(circle)
  mesh.plot()

  func = lambda x, y: np.exp(-(x**2 + y**2))

  plot_function_on_mesh(mesh, func)
