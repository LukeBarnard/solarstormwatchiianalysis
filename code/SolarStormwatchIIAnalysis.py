import os
import glob
import pandas as pd
import numpy as np

def get_project_dirs():
    """
    A function to load in dictionary of project directories stored in a config.txt file stored in the
    :return proj_dirs: Dictionary of project directories, with keys 'data','figures','code', and 'results'.
    """
    files = glob.glob('project_directories.txt')

    if len(files) != 1:
        # If too few or too many config files, guess projdirs
        print('Error: Cannot find correct config file with project directories. Assuming typical structure')
        proj_dirs = {'data': os.path.join(os.getcwd(), 'data'), 'figs': os.path.join(os.getcwd(), 'figures'),
                     'code': os.path.join(os.getcwd(), 'code'), 'results': os.path.join(os.getcwd(), 'results'),
                     'hi_data': os.path.join(os.getcwd(), r'STEREO\ares.nrl.navy.mil\lz\L2_1_25'),
                     'out_data': os.path.join(os.getcwd(), 'out_data'),
                     'ssw_data': os.path.join(os.getcwd(), 'data\\solar-stormwatch-ii-classifications.csv')}

    else:
        # Open file and extract
        with open(files[0], "r") as f:
            lines = f.read().splitlines()
            proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}

    # Just check the directories exist.
    for val in proj_dirs.itervalues():
        if not os.path.exists(val):
            print('Error, invalid path, check config: ' + val)

    return proj_dirs


def import_classifications(latest=True, version=None):
    """
    Function to load in the classifications and do some initial selection on them. Defaults to keeping only the
    classifications from the latest version of the workflow.
    :param latest: Bool, True or False depending on whether the classifications from the latest version of the workflow
                   should be returned.
    : param version: Float, version number of the workflow to extract the classifications. Defaults to None. Overrides
                     the "latest" input.
    :return data: A pandas dataframe containing the requested classifications.
    """
    if not isinstance(latest, bool):
        print("Error: latest should be type bool. Defaulting to True")
        latest = True

    if not version is None:
        # Version not none, so should be float.
        if not isinstance(version, float):
            print("Error, version ({}) should be None or a float. Defaulting to None.".format(version))
            version = None

    # Load in all the data files, get dictionary of asset names, and for each store file names/times
    # Use these to look up classifications;
    # Loop through selected classifications and use json to pull out the x,y coordinates in each frame.
    # Is it worth making a python class "asset", in which I can bundle all of seperate users clicks dict of user and points?
    # the a dict of points x:y

    project_dirs = get_project_dirs()
    # Get converters for the dataframe.
    converters = dict(classification_id=int, user_name=str, user_id=int, user_ip=str, workflow_id=int,
                      workflow_name=str, workflow_version=float, created_at=str, gold_standard=str, expert=str,
                      metadata=str, annotations=str, subject_data=str, subject_ids=str)

    # Load the classifications into dataframe and get latest subset if needed.
    data = pd.read_csv(project_dirs['ssw_data'], converters=converters)

    # Convert the subject ids, which are a variable length list of asset ids.
    subject_ids = [int(line['subject_ids'.split(";")][0]) for idx, line in data.iterrows()]
    data['subject_ids'] = subject_ids



    if version is None:
        if latest:
            data = data[ data['workflow_version'] == data['workflow_version'].max() ]
    else:
        # Check the version exists.
        all_versions = set(np.unique(data['workflow_version']))
        if version in all_versions:
            data = data[data['workflow_version'] == version]
        else:
            print("Error: requested version ({}) doesn't exist. Returning all classifications".format(version))

    return data