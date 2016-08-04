import glob
import os
import tables
import astropy.units as u
import hi_processing as hip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson as json
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


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

    if version is not None:
        # Version not none, so should be float.
        if not isinstance(version, float):
            print("Error, version ({}) should be None or a float. Defaulting to None.".format(version))
            version = None

    project_dirs = get_project_dirs()
    # Load in classifications:
    # Get converters for each column:
    converters = dict(classification_id=int, user_name=str, user_id=int, user_ip=str, workflow_id=int,
                      workflow_name=str, workflow_version=float, created_at=str, gold_standard=str, expert=str,
                      metadata=json.loads, annotations=json.loads, subject_data=json.loads, subject_ids=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['classifications'], converters=converters)

    # Convert the subject ids, which are a variable length list of subject_ids. Because of setup of workflow
    # these numbers just repeat. Only need first.
    # Also, the json annotations are buried in a list, but only the dictionary in element 0 is needed. So pull this out
    # too
    subject_ids = []
    annos = []
    for idx, row in data.iterrows():
        subject_ids.append(row['subject_data'].keys()[0])
        annos.append(row['annotations'][0])

    # Set the new columns
    data['subject_id'] = pd.Series(subject_ids, dtype=int, index=data.index)
    data['annotations'] = pd.Series(annos, dtype=object, index=data.index)

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

    # Correct the index in case any have been removed
    data.set_index(np.arange(data.shape[0]), inplace=True)
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
    converters = dict(subject_id=int, project_id=int, workflow_ids=json.loads, subject_set_id=int, metadata=json.loads,
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

    # Correct the index if subset selected
    data.set_index(np.arange(data.shape[0]), inplace=True)
    return data


def get_df_subset(data, col_name, values, get_range=False):
    """
    Function to return a copy of a subset of a data frame:
    :param data: pd.DataFrame, Data from which to take a subset
    :param col_name: Column name to index the DataFrame
    :param values: A list of values to compare the column given by col_name against. If get_range=True, values should be
                   a list of length 2, giving the lower and upper limits of the range to select between. Else, values
                   should be a list of items, where data['col_name'].isin(values) method is used to select data subset.
    :param get_range: Bool, If true data is returned between the two limits provided by items 0 and 1 of values. If
                      False data returned is any row where col_name matches an element in values.
    :return: df_subset: A copy of the subset of values in the DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        print("Error: data should be a DataFrame")

    if not (col_name in data.columns):
        print("Error: Requested col_name not in data columns")

    # Check the values were a list, if not convert
    if not isinstance(values, list):
        print("Error: Values should be a list. Converting to list")
        values = [values]

    if get_range:
        # Range of values requested. In this case values should be a list of length two. Check
        if len(values) != 2:
            print("Error: Range set True, so values should have two items. ")

    # TODO: Add in a check for the dtype of value.

    if get_range:
        # Get the subset between the range limits
        data_subset = data[(data[col_name] >= values[0]) & (data[col_name] <= values[1])].copy()
    else:
        # Get subset from list of values
        data_subset = data[data[col_name].isin(values)].copy()

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


def get_event_subject_tree(active=True):
    """
    Function to return a dictionary providing the distribution of subject id numbers amongst the hierarchy of
    ssw event id number, craft and image type. This dictionary is used to make it easier to select subsets of
    classifications relating to a particular event / craft / image type.
    :param active: Bool, True if the dictionary should only return information on subjects currently assigned to an
                   active workflow
    :return event_tree: A dictionary with the structure event_tree[event_key][craft][im_type] = [subject_ids]
    """
    subjects = import_subjects(active=active)

    event_ids = []

    for ids, sub in subjects.iterrows():
        ssw_code = sub['metadata']['subject_name'].split('_')
        ssw_code = "_".join(ssw_code[0:2])
        event_ids.append(ssw_code)

    # Get unique event id's
    event_ids = list(set(event_ids))

    event_tree = dict()
    for e in event_ids:
        event_tree[e] = {'sta': {'norm': [], 'diff': []}, 'stb': {'norm': [], 'diff': []}}

    # Now populate the lists in event_num/craft/im_type lists with the relevant subject ids
    for ids, sub in subjects.iterrows():
        name = sub['metadata']['subject_name'].split('_')
        ssw_code = "_".join(name[0:2])
        craft = name[4]
        im_type = name[5]

        event_tree[ssw_code][craft][im_type].append(sub['subject_id'])

    return event_tree


def match_all_classifications_to_ssw_events(active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data structure containing linking each
    classification with it's corresponding frame.
    :param active: Bool, If true only processes subjects currently assigned to a workflow.
    :param latest: Bool, If true only processes classifications from the latest version of the workflow
    :return:
    """

    project_dirs = get_project_dirs()

    # Get classifications from latest workflow
    all_classifications = import_classifications(latest=latest)
    # Only import currently active subjects
    all_subjects = import_subjects(active=active)
    # Get the event tree, detailing which subjects relate to which event number/ craft and image type.
    event_tree = get_event_subject_tree(active=active)

    # Setup the HDF5 data file.
    # Clear it out if it already exists:
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')
    if os.path.exists(ssw_out_name):
        os.remove(ssw_out_name)

    # PyTables Table definition for a users pixels
    class user_pixel_coords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer

    # PyTables Table definition for all users pixels, with added weight, based on number of points in users profile.
    class all_pixel_coords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        weight = tables.Int32Col()  # 32-bit integer

    # Create the hdf5 output file
    ssw_out = tables.open_file(ssw_out_name, mode = "w", title = "Test file")

    # Loop through the event tree, making relevant groups in the ssw_out file.
    # Group structure:

    for event_key, craft in event_tree.iteritems():
        event_group = ssw_out.create_group("/", event_key)

        for craft_key, im_type in craft.iteritems():
            craft_group = ssw_out.create_group(event_group, craft_key)

            for im_key, subjects in im_type.iteritems():
                im_group = ssw_out.create_group(craft_group, im_key)

                for sub in subjects:
                    # Get the subjects for this subject_id
                    subjects_subset = get_df_subset(all_subjects, 'subject_id', [sub], get_range=False)
                    # Meta info needed for making asset names
                    meta = subjects_subset['metadata'].values[0]
                    # Get the classifications for this subject subset
                    clas_subset = get_df_subset(all_classifications, 'subject_id', [sub], get_range=False)

                    # Loop over the assets in this subject
                    for asset_id in range(3):
                        asset_key = "asset_{}".format(asset_id)
                        # Form a timestring for the hdf5 branch name
                        file_parts = os.path.splitext(meta[asset_key])[0].split('_')
                        time_string = "T" + "_".join(file_parts[6:8])
                        time_string_iso = pd.datetime.strptime(time_string, 'T%Y%m%d_%H%M%S').isoformat()

                        # Make a group for this asset
                        asset_group = ssw_out.create_group(im_group, time_string, title=time_string_iso)

                        # Form lists for storing all of the user classifications, id numbers and weight
                        all_x_pix = []
                        all_y_pix = []
                        all_user_weight = []
                        all_user = []

                        # Counter for the number of submissions on this asset
                        submission_count = 0
                        # Iterate through classification rows to find annotations on this asset
                        for idc, cla in clas_subset.iterrows():
                            # Loop through values and find value with frame matching asset index
                            for val in cla['annotations']['value']:
                                # Does this frame match the asset of interest?
                                if val['frame'] == asset_id:
                                    # On first pass set up a users group
                                    if submission_count == 0:
                                        users_group = ssw_out.create_group(asset_group, 'users')

                                    # Create a group for this user
                                    user_key = "user_{}".format(submission_count)
                                    user = ssw_out.create_group(users_group, user_key)
                                    # Make a table for this users annotation
                                    table = ssw_out.create_table(user, 'pixel_coords', user_pixel_coords, title="Pixel coords")
                                    # Assign the annotation to the table
                                    coord = table.row
                                    for p in val['points']:
                                        coord['x'] = int(p['x'])
                                        coord['y'] = 1023-int(p['y'])
                                        coord.append()
                                    # Write to file.
                                    table.flush()

                                    # Dump users data into collection of annotations.
                                    # Get the data for this user classification
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [1023 - int(p['y']) for p in val['points']]
                                    user_weight = np.zeros(len(user_x_pix), dtype=int) + len(user_x_pix)
                                    user_id = cla['user_id']
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    all_user_weight.extend(user_weight)
                                    all_user.append(user_id)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        table = ssw_out.create_table(asset_group, 'pixel_coords', all_pixel_coords, title="Pixel coords")
                        # Assign all elements to this row
                        coord = table.row
                        for x, y, w in zip(all_x_pix, all_y_pix, all_user_weight):
                            coord['x'] = x
                            coord['y'] = y
                            coord['weight'] = w
                            coord.append()
                        # Write to file.
                        table.flush()

    ssw_out.close()


def match_user_classifications_to_ssw_events(username, active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data structure containing linking each
    classification of a specific user with it's corresponding frame. This is achieved using PyTables.
    :param username: String, zooniverse username of the user to extract the stormwatch annotations for
    :param active: Bool, If true only processes subjects currently assigned to a workflow.
    :param latest: Bool, If true only processes classifications from the latest version of the workflow
    :return:
    """

    project_dirs = get_project_dirs()

    # Get classifications from latest workflow
    all_classifications = import_classifications(latest=latest)

    # Get subset of these classifications for this user.
    user_classifications = get_df_subset(all_classifications, 'user_name', username, get_range=False)

    # Only import currently active subjects
    all_subjects = import_subjects(active=active)
    # Get the event tree, detailing which subjects relate to which event number/ craft and image type.
    event_tree = get_event_subject_tree(active=active)

    # Setup the HDF5 data file.
    # Clear it out if it already exists:
    user_tag = "_".join(["user", username, "classifications_matched_ssw_events.hdf5"])
    ssw_out_name = os.path.join(project_dirs['out_data'], user_tag)
    if os.path.exists(ssw_out_name):
        os.remove(ssw_out_name)

    # PyTables Table definition for a users pixels
    class user_pixel_coords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer

    # PyTables Table definition for all users pixels, with added weight, based on number of points in users profile.
    class all_pixel_coords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        weight = tables.Int32Col()  # 32-bit integer

    # Create the hdf5 output file
    ssw_out = tables.open_file(ssw_out_name, mode = "w", title = "Test file")

    # Loop through the event tree, making relevant groups in the ssw_out file.
    # Group structure:

    for event_key, craft in event_tree.iteritems():
        event_group = ssw_out.create_group("/", event_key)

        for craft_key, im_type in craft.iteritems():
            craft_group = ssw_out.create_group(event_group, craft_key)

            for im_key, subjects in im_type.iteritems():
                im_group = ssw_out.create_group(craft_group, im_key)

                for sub in subjects:
                    # Get the subjects for this subject_id
                    subjects_subset = get_df_subset(all_subjects, 'subject_id', [sub], get_range=False)
                    # Meta info needed for making asset names
                    meta = subjects_subset['metadata'].values[0]
                    # Get the classifications for this subject subset
                    clas_subset = get_df_subset(user_classifications, 'subject_id', [sub], get_range=False)

                    # Loop over the assets in this subject
                    for asset_id in range(3):
                        asset_key = "asset_{}".format(asset_id)
                        # Form a timestring for the hdf5 branch name
                        file_parts = os.path.splitext(meta[asset_key])[0].split('_')
                        time_string = "T" + "_".join(file_parts[6:8])
                        time_string_iso = pd.datetime.strptime(time_string, 'T%Y%m%d_%H%M%S').isoformat()

                        # Make a group for this asset
                        asset_group = ssw_out.create_group(im_group, time_string, title=time_string_iso)

                        # Form lists for storing all of the user classifications, id numbers and weight
                        all_x_pix = []
                        all_y_pix = []
                        all_user_weight = []
                        all_user = []

                        # Counter for the number of submissions on this asset
                        submission_count = 0
                        # Iterate through classification rows to find annotations on this asset
                        for idc, cla in clas_subset.iterrows():
                            # Loop through values and find value with frame matching asset index
                            for val in cla['annotations']['value']:
                                # Does this frame match the asset of interest?
                                if val['frame'] == asset_id:
                                    # On first pass set up a users group
                                    if submission_count == 0:
                                        clas_group = ssw_out.create_group(asset_group, 'classifications')

                                    # Create a group for this user
                                    clas_key = "classification_{}".format(submission_count)
                                    clas = ssw_out.create_group(clas_group, clas_key)
                                    # Make a table for this users annotation
                                    table = ssw_out.create_table(clas, 'pixel_coords', user_pixel_coords, title="Pixel coords")
                                    # Assign the annotation to the table
                                    coord = table.row
                                    for p in val['points']:
                                        coord['x'] = int(p['x'])
                                        coord['y'] = 1023-int(p['y'])
                                        coord.append()
                                    # Write to file.
                                    table.flush()

                                    # Dump users data into collection of annotations.
                                    # Get the data for this user classification
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [1023 - int(p['y']) for p in val['points']]
                                    user_weight = np.zeros(len(user_x_pix), dtype=int) + len(user_x_pix)
                                    user_id = cla['user_id']
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    all_user_weight.extend(user_weight)
                                    all_user.append(user_id)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        table = ssw_out.create_table(asset_group, 'pixel_coords', all_pixel_coords, title="Pixel coords")
                        # Assign all elements to this row
                        coord = table.row
                        for x, y, w in zip(all_x_pix, all_y_pix, all_user_weight):
                            coord['x'] = x
                            coord['y'] = y
                            coord['weight'] = w
                            coord.append()
                        # Write to file.
                        table.flush()

    ssw_out.close()


def test_plot():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')

    # Load in the event tree dictionary to easily iterate over the events/craft/img type
    ssw_event_tree = get_event_subject_tree(active=True)

    # get handle to the ssw out data
    ssw_out = tables.open_file(ssw_out_name, mode="r")

    # Loop over events, craft and image type
    for event_k, event in ssw_event_tree.iteritems():

        for craft_k, craft in event.iteritems():

            # Make a plot for each image type.
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            for i, img_k in enumerate(craft.iterkeys()):

                # Get path to this node of the data, pull out the group of times
                path = "/".join(['', event_k, craft_k, img_k])
                times = ssw_out.get_node(path)
                # Set up the normalisation and colormap for the plots
                norm = mpl.colors.Normalize(vmin=0, vmax=len(times._v_groups))
                cmap = mpl.cm.viridis
                for frame_count, t in enumerate(times):
                    coords = pd.DataFrame.from_records(t.pixel_coords.read())
                    ax[i].plot(coords['x'], coords['y'], 'o', color=cmap(norm(frame_count)))

                # Label it up and format
                ax[i].set_title("Tracked in {0} {1} image".format(craft_k, img_k))
                ax[i].set_xlim(0, 1023)
                ax[i].set_ylim(0, 1023)
                ax[i].set_aspect('equal')

            # Save and move to next plot
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
            name = os.path.join(project_dirs['figs'], "_".join([event_k, craft_k, 'test.jpg']))
            plt.savefig(name)
            plt.close('all')


def test_animation():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')

    # Load in the event tree dictionary to easily iterate over the events/craft/img type
    ssw_event_tree = get_event_subject_tree(active=True)

    # get handle to the ssw out data
    ssw_out = tables.open_file(ssw_out_name, mode="r")

    # Loop over events and craft
    for event_k, event in ssw_event_tree.iteritems():

        for craft_k, craft in event.iteritems():
            # Make a plot for each image type.
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            # Get path to the normal and diff img types of this event, pull out the group of times
            norm_path = "/".join(['', event_k, craft_k, 'norm'])
            norm_times = ssw_out.get_node(norm_path)

            diff_path = "/".join(['', event_k, craft_k, 'diff'])
            diff_times = ssw_out.get_node(diff_path)

            # Set up the normalisation and colormap for the plots
            norm = mpl.colors.Normalize(vmin=0, vmax=len(diff_times._v_groups))
            cmap = mpl.cm.viridis
            for frame_count, (norm_t, diff_t) in enumerate(zip(norm_times, diff_times)):
                norm_coords = pd.DataFrame.from_records(norm_t.pixel_coords.read())
                ax[0].plot(norm_coords['x'], norm_coords['y'], 'o', color=cmap(norm(frame_count)))

                diff_coords = pd.DataFrame.from_records(diff_t.pixel_coords.read())
                ax[1].plot(diff_coords['x'], diff_coords['y'], 'o', color=cmap(norm(frame_count)))

                # Label it up and format
                for a,img_k in zip(ax, ["norm", "diff"]):
                    a.set_title("Tracked in {0} {1} image".format(craft_k, img_k))
                    a.set_xlim(0, 1023)
                    a.set_ylim(0, 1023)
                    a.set_aspect('equal')

                # Save and move to next plot
                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
                name = "_".join([event_k,  craft_k, "ani_f{0:03d}.jpg".format(frame_count)])
                name = os.path.join(project_dirs['figs'], name )
                plt.savefig(name)

            plt.close('all')

        # Make animation and clean up.
        for craft in ['sta', 'stb']:
            src = os.path.join(project_dirs['figs'], event_k + "_" + craft + "*_ani*.jpg")
            dst = os.path.join(project_dirs['figs'], event_k + "_" + craft + "_front_id_test.gif")
            cmd = " ".join(["convert -delay 0 -loop 0 -resize 50%", src, dst])
            os.system(cmd)
            # Tidy.
            files = glob.glob(src)
            for f in files:
                os.remove(f)


def test_front_density():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')

    # Load in the event tree dictionary to easily iterate over the events/craft/img type
    ssw_event_tree = get_event_subject_tree(active=True)

    # get handle to the ssw out data
    ssw_out = tables.open_file(ssw_out_name, mode="r")

    x = np.arange(0,1024)
    y = np.arange(0, 1024)
    xm, ym = np.meshgrid(x, y)
    all_coords = np.vstack([xm.ravel(), ym.ravel()]).T

    # Loop over events and craft
    for event_k, event in ssw_event_tree.iteritems():

        if event_k != "ssw_009":
            continue

        for craft_k, craft in event.iteritems():

            # Get path to the normal and diff img types of this event, pull out the group of times
            norm_path = "/".join(['', event_k, craft_k, 'norm'])
            norm_times = ssw_out.get_node(norm_path)

            diff_path = "/".join(['', event_k, craft_k, 'diff'])
            diff_times = ssw_out.get_node(diff_path)

            # Set up the normalisation and colormap for the plots
            norm = mpl.colors.Normalize(vmin=0, vmax=len(diff_times._v_groups))
            cmap = mpl.cm.viridis
            for frame_count, (norm_t, diff_t) in enumerate(zip(norm_times, diff_times)):
                # Get the normal and diff coords
                norm_coords = pd.DataFrame.from_records(norm_t.pixel_coords.read())
                diff_coords = pd.DataFrame.from_records(diff_t.pixel_coords.read())
                xy = np.vstack([norm_coords['x'].values.ravel(), norm_coords['y'].values.ravel()]).T
                # Fit a KDE to these points
                # 20-fold cross-validation
                if norm_coords.shape[0] < 20:
                    continue

                grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(1, 10.0, 30)})
                grid.fit(xy)
                print grid.best_params_
                kde = grid.best_estimator_
                pdf = np.reshape(kde.score_samples(all_coords),xm.shape)

                # Mark on the points
                fig, ax = plt.subplots(1, 2, figsize=(15, 7))
                ax[0].contourf(xm, ym, pdf, cmap=cmap)
                ax[1].contourf(xm, ym, pdf, cmap=cmap)
                ax[0].plot(norm_coords['x'], norm_coords['y'], 'o', color='r')
                ax[1].plot(diff_coords['x'], diff_coords['y'], 'o', color='r')
                plt.show()
                # Label it up and format
                for a, img_k in zip(ax, ["norm", "diff"]):
                    a.set_title("Tracked in {0} {1} image".format(craft_k, img_k))
                    a.set_xlim(0, 1023)
                    a.set_ylim(0, 1023)
                    a.set_aspect('equal')

                # Save and move to next plot
                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
                name = "_".join([event_k, craft_k, "ani_f{0:03d}.jpg".format(frame_count)])
                name = os.path.join(project_dirs['figs'], name)
                plt.savefig(name)
                plt.close('all')
            break
        break
        # Make animation and clean up.
        for craft in ['sta', 'stb']:
            src = os.path.join(project_dirs['figs'], event_k + "_" + craft + "*_ani*.jpg")
            dst = os.path.join(project_dirs['figs'], event_k + "_" + craft + "_front_id_test.gif")
            cmd = " ".join(["convert -delay 0 -loop 0 -resize 50%", src, dst])
            os.system(cmd)
            # Tidy.
            files = glob.glob(src)
            for f in files:
                os.remove(f)


def test_front_reconstruction():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')

    hdf = h5py.File(out_hdf5_name, "r")
    ssw_event = 'ssw_009'
    craft = 'sta'
    im_type = 'diff'
    path = "/".join(['', ssw_event, craft, im_type])
    data = hdf[path]

    # Use data keys to get times, then make time pa DF to store front elongation in.

    # Loop through the times.
    # >>> Find relevant fits file. - done
    # >>> Get HPR coordinates. - done
    # >>> Convert the pixel coordinates to HPR coordinates. - done
    # >>> Loop through PA grid - done
    # >>> ^^^ Find points in this PA window. - done
    # >>> ^^^ KS density fit to elongation of these points. - done
    # >>> ^^^ Find elongation of maximum of estimated dist. - done
    # >>> ^^^ Append to DF - done
    # >>> Use Storm front DF to add in pa,elongation and pixel coords of the Stormfront to the HDF dataframe.
    # >>> Should I spline gaps in the storm front profile? In HPR coords this may not be a bad fit, as pretty linear.

    # Get pa bin
    pa_bin_wid = 5
    pa_bins = np.arange(50, 130, pa_bin_wid)
    pa_tol = pa_bin_wid / 2.0
    # El values to loop up dist at.
    el_bins = np.arange(4, 25, 0.1)

    columns  = {t:pd.Series(data=np.zeros(pa_bins.size)) for t in data.iterkeys()}
    storms = pd.DataFrame(data=columns, index=pa_bins)
    del columns

    for t, anno in data.iteritems():

        t_start = pd.to_datetime(t)
        files = hip.find_hi_files(t_start, t_start, craft=craft, camera='hi1', background_type=1 )
        files = files[0]
        print files
        himap = hip.get_image_plain(files, star_suppress=False)

        # Convert classifications from pixels to to HPR coords.
        x = np.array(anno['x_pix'][...]) * u.pix
        y = np.array(anno['y_pix'][...]) * u.pix
        el, pa = hip.convert_pix_to_hpr(x, y, himap)

        # Loop through position angles, find points in that bin, work out elongation distribution, get mode.
        for pab in pa_bins:
            el_sub = el[ (pa.value >= pab - pa_tol) & (pa.value <= pab + pa_tol)]
            if len(el_sub) >= 5:
                kernal = stats.gaussian_kde(el_sub.value)
                el_dist = kernal(el_bins)
                el_consensus = el_bins[np.argmax(el_dist)]
                # plt.figure()
                # plt.plot(el_bins, el_dist,'k-')
                # plt.plot(el_sub, np.zeros(el_sub.size), 'r.')
                # ylims = plt.gca().get_ylim()
                # plt.vlines(el_consensus, ylims[0], ylims[1], colors='b')
                # plt.ylim(ylims[0], ylims[1])
                # plt.show()
            else:
                el_consensus = np.NaN

            # Assign to storms
            storms.loc[pab,t] = el_consensus

