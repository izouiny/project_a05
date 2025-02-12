# NHL Goal Prediction Engine - Milestones 1 and 2 of Data Science course (IFT6758)

This is the code for the first and second milestones of this project. The former includes exploration and visualization of the data. The latter is about feature engineering, feature selection, model selection and hyperparameters' optimization for predicting if a shot will be a goal. The third milestone is in another repository, right [here](https://github.com/PhilSchoeb/NHL_milestone3). The analysis of our work for the two first milestones can be found at [this repository](https://github.com/PhilSchoeb/NHL_blog_milestone1-2).

# IFT6758 Repo Template
 
This template provides you with a skeleton of a Python package that can be installed into your local machine.
This allows you access your code from anywhere on your system if you've activated the environment the package was installed to.
You are encouraged to leverage this package as a skeleton and add all of your reusable code, functions, etc. into relevant modules.
This makes collaboration much easier as the package could be seen as a "single source of truth" to pull data, create visualizations, etc. rather than relying on a jumble of notebooks.
You can still run into trouble if branches are not frequently merged as work progresses, so try to not let your branches diverge too much!

Also included in this repo is an image of the NHL ice rink that you can use in your plots.
It has the correct location of lines, faceoff dots, and length/width ratio as the real NHL rink.
Note that the rink is 200 feet long and 85 feet wide, with the goal line 11 feet from the nearest edge of the rink, and the blue line 75 feet from the nearest edge of the rink.

<p align="center">
<img src="./figures/nhl_rink.png" alt="NHL Rink is 200ft x 85ft." width="400"/>
<p>

The image can be found in [`./figures/nhl_rink.png`](./figures/nhl_rink.png).

## Installation

To install this package, first setup your Python environment by following the instructions in the [Environment](#environments) section.
Once you've setup your environment, you can install this package by running the following command from the root directory of your repository. 

    pip install -e .

You should see something similar to the following output:

    > pip install -e .
    Obtaining file:///home/USER/project-template
    Installing collected packages: ift6758
    Running setup.py develop for ift6758
    Successfully installed ift6758-0.1.0


## Environments

The first thing you should setup is your isolated Python environment.
You can manage your environments through either Conda or pip.
Both ways are valid, just make sure you understand the method you choose for your system.
It's best if everyone on your team agrees on the same method, or you will have to maintain both environment files!
Instructions are provided for both methods.

**Note**: If you are having trouble rendering interactive plotly figures and you're using the pip + virtualenv method, try using Conda instead.

### Conda 

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate ift6758-project

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-project

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Pip + Virtualenv

An alternative to Conda is to use pip and virtualenv to manage your environments.
This may play less nicely with Windows, but works fine on Unix devices.
This method makes use of the `requirements.txt` file; you can disregard the `environment.yml` file if you choose this method.

Ensure you have installed the [virtualenv tool](https://virtualenv.pypa.io/en/latest/installation.html) on your system.
Once installed, create a new virtual environment:

    vitualenv ~/ift6758-venv
    source ~/ift6758-venv/bin/activate

Install the packages from a requirements.txt file:

    pip install -r requirements.txt

As before, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=ift6758-venv

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you want to create a new `requirements.txt` file, you can use `pip freeze`:

    pip freeze > requirements.txt



