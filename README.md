# cMRI-Voxel2Mesh
Multiclass implementation of the Voxel2Mesh architecture with mesh reconstruction of cardiac MRI image volumes & CAT segmentation.

## 1. Environment Setup
When I've set this up on different PCs, I've made my own conda environment and manually added the packages below... I found
this easier and less hassle than trying to share an environment file, especially since the versions of Cuda and PyTorch
are dependent on your PC/GPU.
### Packages:
- Python 3.9
- pynrrd 1.1.3
- Cuda 12.x
- PyTorch 2.x
- torch-geometric 2.6.1
- numpy 2.0.1
- trimesh 4.6.5
- scipy 1.13.1
- skimage 0.24.0
- sklearn 1.6.1
- pyvista
- matplotlib

## 2. Training a Model
There's a few key things to keep in mind when training a new model. If you've set everything up according to the following
steps/points, all you need to do is run main.py.

### - Data Setup
Setting up the data properly is super important! Unfortunately, some of our data labels either don't work properly, or 
some are structured differently, such that my model can't run them.  The following are one's that are OK enough to use:
- MF0509-ED, ES
- MF0510-ED, ES
- MF0512-ES
- MF0514-ED, ES
- MF0515-ED, ES
- MF0516-ED, ES
- MF0518-ED, ES
- MF0519-ED, ES

Hopefully we can add to this list as the gradient echos are traced or if we redo some of the other scans :)  

              -> train  -> images, labels    
       
       data ->

              -> test   -> images, labels
       
### - Warmup Epochs 
The variable 'warmup_epochs' in main.py is the number of epochs that will be run ONLY on segmentation... this is key since 
we want the segmentations to be decent before we start trying to deform the template mesh.  If warmup_epochs is too small, 
the marching cubes of the segmentations can cause errors.

### - Label Extraction
Extracting labels from the segmentation volume happens in the 'load_labels()' function in data.py.  This has a lot of 
implications regarding the training process - essentially we're structuring our own label volume in this function.  
Depending on which labels we want to group together and what order we have them in, there's several things you have to
change.  
Fat should ALWAYS be the last label.  Depending on which labels you have, you may have to change colormaps and labels
across the visualization functions in model.py and validation.py.

### - Config.py
Config.py holds a lot of the info that's referenced throughout this project.  If you change the label volume, make sure
you change the corresponding information in this file, including 'nrrd_dimensions', 'num_classes', 'num_mesh_classes'.

### - Losses
Controlling the loss weights (in model.py) is really sensitive.  Speaking from experience, I'd take notes of the weights that tend to
work the best, and slightly alter them based on the results you get.  
Changing the weighting of the classes' cross entropy loss is another thing that will vastly change how the model operates.
Due to the class imbalance in the number of voxels for each class, if you change the structure of your label volume,
you'll __definitely__ need to change these weights (in losses.py, cross entropy loss function).

### - Number of points
The number of mesh points in the training process pretty much dictates how fast the training will be, but also controls the
resolution for the mesh.  I'd recommend going with the size 2562 sphere, and using double this amount (5134), because
of Nyquist :)

## 3. Running / Testing a Model
Validation.py holds a lot of useful code to run/test a model.  I've added a bunch of functions that visualize segmentation
slices, as well as plot each chamber's mesh.  Again, if you change the label volume's structure, you'll need to change
some of the visualization parameters to match this.  After you've got everything set up, running validation.py will plot
all of these, as well as save a text file that holds some of the results.

