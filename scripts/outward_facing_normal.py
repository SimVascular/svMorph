import vtk

# Load the .vtp file
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("/home/bohanjeffli/mesh-complete-exterior.vtp")

# Compute cell (face) normals
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(reader.GetOutputPort())
normals.ComputePointNormalsOff()
normals.ComputeCellNormalsOn()
normals.ConsistencyOn()
normals.AutoOrientNormalsOn()
normals.SplittingOff()
normals.Update()

# Get cell centers to place the arrows
centers = vtk.vtkCellCenters()
centers.SetInputConnection(normals.GetOutputPort())
centers.Update()

# Arrow glyph
arrowSource = vtk.vtkArrowSource()

# Glyph arrows at face centers using face normals
glyph = vtk.vtkGlyph3D()
glyph.SetSourceConnection(arrowSource.GetOutputPort())
glyph.SetInputConnection(centers.GetOutputPort())
glyph.SetVectorModeToUseNormal()
glyph.SetScaleModeToScaleByVector()
glyph.SetScaleFactor(0.1)  # Adjust based on mesh size
glyph.OrientOn()
glyph.Update()

# Mapper and actor for original mesh
mesh_mapper = vtk.vtkPolyDataMapper()
mesh_mapper.SetInputConnection(normals.GetOutputPort())

mesh_actor = vtk.vtkActor()
mesh_actor.SetMapper(mesh_mapper)
mesh_actor.GetProperty().SetOpacity(0.3)  # translucent mesh

# Mapper and actor for glyphs
glyph_mapper = vtk.vtkPolyDataMapper()
glyph_mapper.SetInputConnection(glyph.GetOutputPort())

glyph_actor = vtk.vtkActor()
glyph_actor.SetMapper(glyph_mapper)
glyph_actor.GetProperty().SetColor(1, 0, 0)  # red arrows

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(mesh_actor)
renderer.AddActor(glyph_actor)
renderer.SetBackground(1, 1, 1)

# Render window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.SetRenderWindow(renderWindow)

# Start interaction
renderWindow.Render()
interactor.Start()