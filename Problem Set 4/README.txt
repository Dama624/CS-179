Question 1: Parallel Breadth-First Search (BFS)
--------------------------------------------------------
--------------------------------------------------------

1.1
---------------------
It would not.
To take full advantage of shared memory, we need to read from the shared memory multiple times (since we want to minimize the number of times we write from global memory to shared memory). To maximize the number of times we read from shared memory, we would need to make sure the “layers” (and its nodes) of the graph we are reading from are located relatively near each other. However, figuring this out is exactly the problem we are trying to solve - the distance between the source and nodes of a graph.

1.2
---------------------
Perform a reduction (such as the one done for Problem Set 3). After the reduction, perform an Atomic Add between all indices of F, then at the end of the loop, check if the sum of the indices is 0. We do another iteration if the sum does not equal 0.

1.3
---------------------
Initialize a global boolean variable; set it to 1 when F[j] is set to true. Set it to 0 when F[threadId] is true. F is not all false for as long as this variable is set to 1.
This change has a notable disadvantage in dense graphs, since it is writing to this global variable for each node within a “layer”. In dense graphs, this means there are many nodes within a single layer, and so there are many more writes to this variable. In fact, it is possible that it could write more times to this variable then there are nodes in the graph (in which case checking if every element of F is false would have better performance).
In sparse graphs, however, writing to this variable may result in better performance, especially in cases where some branches of the paths result in a dead end (and so do not require a write to this variable again).


Question 2: Algorithm compare/contrast: PET Reconstruction
--------------------------------------------------------
--------------------------------------------------------
We are parallelizing over each measurement rather than pixel, so our PET reconstruction performance will be greatly reduced compared to that of the X-ray CT.
For one, we are parallelizing over each measurement. Because each measurement corresponds to a linear line of pixels through the image, we already face reduced performance since each “parallelized line” is performing work for multiple pixels (versus the CT reconstruction, where each pixel is its own thread). 
For another, we are unable to use texture memory for caching sinogram reads. We could do so for CT because the CT sinogram reads were comprised of **unique** thetas and “distance”s. For the PET, however, multiple reads can be measured at the same location. There will be non-coalesced access to the sinogram for the PET.
