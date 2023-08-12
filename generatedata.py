

import ufl
from dolfinx import fem, io, mesh, plot
import dolfinx
from ufl import ds, dx, grad, inner
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import numpy as np
from tqdm import trange

def compute(i):
    mu=1.0+4.0*i/100
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1, 1)), n=(255, 255),
                            cell_type=mesh.CellType.quadrilateral)
    V = fem.FunctionSpace(msh, ("Lagrange", 1))
    facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 1)),np.logical_or(np.isclose(x[1], 0.0),
                                                                      np.isclose(x[1], 1))))
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1,entities=facets)
    bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx + mu*inner(u, v) * dx
    x = ufl.SpatialCoordinate(msh)
    f =mu*((x[0] - 1/2) ** 2 + (x[1] - 1/2) ** 2) 
    L = inner(f, v) * dx 
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    fdim = msh.topology.dim - 1
    num_facets_owned_by_proc = msh.topology.index_map(fdim).size_local
    geometry_entitites = dolfinx.cpp.mesh.entities_to_geometry(msh, fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)
    points = msh.geometry.x
    u_val=uh.x.array
    #with io.XDMFFile(msh.comm, "simulations/"+str(i)+".xdmf", "w") as xdmf:
    #        xdmf.write_mesh(msh)
    #        xdmf.write_function(uh)
    return u_val,points,mu

outputs=np.zeros((100,256,256))
inputs=np.zeros((100,1))

for i in trange(100):
  u_val,points,mu=compute(i)  
  points=np.array(points*255,dtype=np.int64)
  inputs[i]=mu
  for j in range(len(points)):
    outputs[i,points[j,0],points[j,1]]=u_val[j]+1

np.save("inputs.npy",inputs)
np.save("outputs.npy",outputs)







import matplotlib.pyplot as plt
import numpy as np
from torch import nn
inputs=np.load("inputs.npy")
outputs=np.load("outputs.npy")


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.nn=nn.Sequential(nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2),nn.ReLU(),nn.Dropout(0.5),
                              nn.ConvTranspose2d(1,1,2,2))
        
    def forward(self,x):
        x=x.reshape(x.shape[0],1,1,1)
        return self.nn(x).squeeze(1)