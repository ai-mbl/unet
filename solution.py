# %% [markdown]
"""
# Example Exercise
This is an example exercise notebook written as a python file and converted to a notebook using jupytext and nbconvert.

Please start your exercise with a brief description of the learning objectives and task in the notebook. E.g. in this notebook we are going to train a simple linear classifier to identify if an image contains a dog or a cat. At the end of this exercise, you should be familiar with how to use tensorflow to implement a simple model and how to set up a dataset object.

<div class="alert alert-danger">
Set your python kernel to <code>000-example-exercise</code>
</div>
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
Tasks should be clearly numbered and contain an explicit piece of work that the students need to complete.

<div class="alert alert-info">

### Task 1.1

Use <code>np.zeros</code> to generate an empty array of zeros with the shape 10x10x2. For more information on this function, check out the <a href="https://numpy.org/doc/stable/reference/generated/numpy.zeros.html">docs</a> or run <code>np.zeros?</code> in a code cell to see the docstring.
</div>
"""

# %% tags=["task"]
a = ...  # Insert your code for generating an array of zeros here

# Check that the shape is correct
a.shape

# %% [markdown]
# Solution cells are tagged with "solution" and can be removed using `nbconvert`. See the cell below as an example

# %% tags=["solution"]
a = np.zeros((10, 10, 2))

a.shape

# %% [markdown]
"""
Checkpoints help us keep track of the students progress and make sure they are moving through the exercise appropriately. You should aim for a checkpoint for each hour of work or each conceptual block. They are a great opportunity to make sure the students understand what has been covered so far and are prepared to move on to the next section.
<div class="alert alert-success">

## Checkpoint 1

Post to Element when you have reached checkpoint 1. We will discuss the many wonderful uses of <code>np.zeros</code>.
</div>

<div class="alert alert-warning">

### Question
This box poses a question for students to ponder.
</div>
"""
