import os
import glob
import pandas as pd
import numpy as np
import simplejson as json
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt


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
                     'classifications': os.path.join(os.getcwd(), 'data\\solar-stormwatch-ii-classifications.csv'),
                     'subjects': os.path.join(os.getcwd(), 'data\\solar-stormwatch-ii-subjects.csv'),
                     'workflow_contents': os.path.join(os.getcwd(), 'data\\solar-stormwatch-ii-workflow_contents.csv'),
                     'workflows': os.path.join(os.getcwd(), 'data\\solar-stormwatch-ii-workflows.csv')}

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
    :param version: Float, version number of the workflow to extract the classifications. Defaults to None. Overrides
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

    project_dirs = get_project_dirs()
    # Load in classifications:
    # Get converters for each column:
    converters = dict(classification_id=int, user_name=str, user_id=int, user_ip=str, workflow_id=int,
                      workflow_name=str, workflow_version=float, created_at=str, gold_standard=str, expert=str,
                      metadata=str, annotations=str, subject_data=str, subject_ids=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['classifications'], converters=converters)
    # Convert the subject ids, which are a variable length list of subject_ids. Because of setup of workflow
    # these numbers just repeat. Only need first.
    subject_ids = []
    for idx, line in data.iterrows():
        info = json.loads(line['subject_data'])
        subject_ids.append(info.keys()[0])

    data['subject_id'] = pd.Series(subject_ids, dtype=int, index=data.index)
    # Drop the old subject_ids, as not needed
    data.drop('subject_ids', axis=1, inplace=True)

    if version is None:
        if latest:
            data = data[data['workflow_version'] == data['workflow_version'].tail(1).values[0]]
    else:
        # Check the version exists.
        all_versions = set(np.unique(data['workflow_version']))
        if version in all_versions:
            data = data[data['workflow_version'] == version]
        else:
            print("Error: requested version ({}) doesn't exist. Returning all classifications".format(version))

    return data


def import_subjects(active=False):
    """
    Function to load in the classifications and do some initial selection on them. Defaults to retrieving all subjects,
    but can optionally retrieve only subjects associated with workflows.
    :param active: Bool, True or False depending on whether the only subjects assigned to a workflow should be selected.
    :return data: A pandas DataFrame containing the requested classifications.
    """
    if not isinstance(active, bool):
        print("Error: active should be type bool. Defaulting to False")
        active = False

    project_dirs = get_project_dirs()
    # Load in classifications:
    # Get converters for each column:
    converters = dict(subject_id=int, project_id=int, workflow_ids=json.loads, subject_set_id=int, metadata=str,
                      locations=str, classifications_by_workflow=str, retired_in_workflow=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['subjects'], converters=converters)
    # Sort out the workflow ids which are in an awkward format

    if active:
        ids = np.zeros(len(data), dtype=bool)
        for idx, d in data.iterrows():
            if not d['workflow_ids'] == []:
                ids[idx] = 1
        data = data[ids]

    return data


def get_df_subset(data, col_name, value):
    """
    Function to return a copy of a subset of a data frame:
    :param data: pd.DataFrame, Data from which to take a subset
    :param col_name: Column name to index the DataFrame
    :param value: Value to compare the column given by col_name against
    :return: df_subset: A copy of the subset of values in the DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        print("Error: data should be a DataFrame")

    if not (col_name in data.columns):
        print("Error: Requested col_name not in data columns")

    # TODO: Add in a check for the dtype of value.

    data_subset = data[data[col_name] == value].copy()
    return data_subset


def get_subject_set_files(subjects, subject_set_ids, get_all=False):
    """
    Function to retrieve the subject ids and filenames in each fileset.
    :param subjects: DataFrame of subjects, from import_subjects.
    :param subject_set_ids: List of subject_set_ids
    :param get_all: Bool. When True overrides subject_set_ids and retrieves the file list for every set in subjects.
    :return:
    """

    if not isinstance(get_all, bool):
        print("Error: get_all should be bool. Defaulting to False")
        get_all = False

    if not isinstance(subjects, pd.DataFrame):
        print("Error: Subjects should be a DataFrame.")

    if not isinstance(subject_set_ids, list):
        print("Error: subject_set_ids should be a list")

    if not subject_set_ids:
        print("Error: subject_set_ids shouldn't be an empty list.")

    if get_all:
        # Override subject_set_ids and get all subject_set_ids
        subject_set_ids = subjects['subject_set_id'].unique()

    subject_files = {}
    for set_id in subject_set_ids:

        subject_part = get_df_subset(subjects, 'subject_set_id', set_id)
        norm_files = []
        norm_sub_id = []
        diff_files = []
        diff_sub_id = []
        for idx, sub in subject_part.iterrows():

            info = json.loads(sub['metadata'])

            if info['img_type'] == 'norm':
                norm_sub_id.append(sub["subject_id"])
            elif info['img_type'] == 'diff':
                diff_sub_id.append(sub["subject_id"])

            for val in info.itervalues():
                if val.endswith(".jpg"):
                    if info['img_type'] == 'norm':
                        norm_files.append(val)
                    elif info['img_type'] == 'diff':
                        diff_files.append(val)

        diff_files.sort()
        norm_files.sort()
        # Get craft info too.
        if 'sta' in info['subject_name']:
            subject_files[set_id] = {'craft': 'sta', 'norm': {'files': norm_files, 'sub_ids': norm_sub_id},
                                     'diff': {'files': diff_files, 'sub_ids': diff_sub_id}}
        elif 'stb' in info['subject_name']:
            subject_files[set_id] = {'craft': 'stb', 'norm': {'files': norm_files, 'sub_ids': norm_sub_id},
                                     'diff': {'files': diff_files, 'sub_ids': diff_sub_id}}

    return subject_files


def create_classification_frame_matched_hdf5(active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data structure containing linking each
    classification with it's corresponding frame.
    :param active: Bool, If true only processes subjects currently assigned to a workflow.
    :param latest: Bool, If true only processes classifications from the latest version of the workflow
    :return:
    """

    project_dirs = get_project_dirs()

    classifications = import_classifications(latest=latest)
    # Only import currently active subjects
    subjects = import_subjects(active=active)

    # Setup the HDF5 data file.
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'out_data.hdf5')
    if os.path.exists(out_hdf5_name):
        os.remove(out_hdf5_name)

    # Create the hdf5 output files
    hdf = h5py.File(out_hdf5_name, "w")

    # Loop through the unique subject sets
    for set_id in subjects['subject_set_id'].unique():
        # Get all subjects in this set
        subject_subset = get_df_subset(subjects, 'subject_set_id', set_id)

        # Get info on the files in this set.
        set_files = get_subject_set_files(subject_subset, [set_id])

        # Create a group for this set
        grp_set = hdf.create_group(str(set_id))
        grp_set.attrs['craft'] = set_files[set_id]['craft']

        # Loop over the subjects in this set
        for ids, sub in subject_subset.iterrows():
            # Get the meta information of assets in the subject
            info = json.loads(sub['metadata'])

            # Find classifications for this subject
            clas_subset = get_df_subset(classifications, 'subject_id', sub['subject_id'])

            # Loop over the assets in this subject
            for asset_id in range(3):
                key = "asset_{}".format(asset_id)
                # Form a timestring for the hdf5 branch name
                file_parts = os.path.splitext(info[key])[0].split('_')
                timestr = "_".join(file_parts[6:8])
                timestr = pd.datetime.strptime(timestr, '%Y%m%d_%H%M%S').isoformat()

                # Make a branch for this image type / asset time.
                branch = "/".join([info['img_type'], timestr])
                #print branch
                grp_asset = grp_set.create_group(branch)

                # Loop over the classifications to pull out the annotations for this asset.
                all_x_pix = []
                all_y_pix = []
                all_user = []
                for j, (idc, cla) in enumerate(clas_subset.iterrows()):

                    # Get the annotation
                    cla_anno = json.loads(cla['annotations'])[0]

                    # Loop through values and find value with frame matching asset index
                    for val in cla_anno['value']:
                        # Does this frame match the asset of interest?
                        if val['frame'] == asset_id:
                            # Get the data for this user classification
                            user_x_pix = [int(p['x']) for p in val['points']]
                            user_y_pix = [int(p['y']) for p in val['points']]
                            user_id = cla['user_id']
                            # Set up a group for this user.
                            branch = "users/user_{}".format(j)
                            #print "---->" + branch
                            grp_user = grp_asset.create_group(branch)
                            # Assign the user data to their group
                            grp_user.create_dataset('x_pix', data=np.array(user_x_pix, dtype=int))
                            grp_user.create_dataset('y_pix', data=np.array(user_y_pix, dtype=int))
                            grp_user.create_dataset('id', data=user_id)
                            # Dump users data into collection of annotations.
                            all_x_pix.extend(user_x_pix)
                            all_y_pix.extend(user_y_pix)
                            all_user.append(user_id)

                # Now add on the collected annotation to the assets group.
                grp_asset.create_dataset('x_pix', data=np.array(all_x_pix, dtype=int))
                grp_asset.create_dataset('y_pix', data=np.array(all_y_pix, dtype=int))
                grp_asset.create_dataset('user_list', data=np.array(all_user, dtype=int))

    hdf.close()


def test_plot():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'out_data.hdf5')

    hdf = h5py.File(out_hdf5_name, "r")
    for grp in hdf.values():
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))

        for im_type, grp2, a in zip(grp.keys(), grp.values(), ax):
            # Get time lims.
            norm = mpl.colors.Normalize(vmin=0, vmax=len(grp2.keys()))
            cmap = mpl.cm.viridis
            c = 0
            for k, v in grp2.iteritems():
                a.plot(v['x_pix'][...], v['y_pix'][...], 'o', color=cmap(norm(c)))
                a.set_title("Tracked in {0} {1} image".format(grp.attrs['craft'], im_type))
                c += 1

        for a in ax:
            a.set_xlim(0, 1023)
            a.set_ylim(0, 1023)
            a.set_aspect('equal')

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
        name = os.path.join(project_dirs['figs'], grp.attrs['craft']+'_test.jpg')
        plt.savefig(name)

