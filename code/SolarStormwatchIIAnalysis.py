import os
import glob
import pandas as pd
import numpy as np
import simplejson as json
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats


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
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')
    if os.path.exists(out_hdf5_name):
        os.remove(out_hdf5_name)

    # Create the hdf5 output files
    hdf = h5py.File(out_hdf5_name, "w")

    for event_key, craft in event_tree.iteritems():

        for craft_key, im_type in craft.iteritems():

            for im_key, subjects in im_type.iteritems():

                for sub in subjects:
                    # Get the subjects for this subject_id
                    subjects_subset = get_df_subset(all_subjects, 'subject_id', [sub], get_range=False)
                    meta = subjects_subset['metadata'].values[0]
                    # Get the classifications for this subset
                    clas_subset = get_df_subset(all_classifications, 'subject_id', [sub], get_range=False)
                    for asset_id in range(3):
                        asset_key = "asset_{}".format(asset_id)
                        # Form a timestring for the hdf5 branch name
                        file_parts = os.path.splitext(meta[asset_key])[0].split('_')
                        time_string = "_".join(file_parts[6:8])
                        time_string = pd.datetime.strptime(time_string, '%Y%m%d_%H%M%S').isoformat()
                        # Make a branch for this image type / asset time.
                        branch = "/".join([event_key, craft_key, im_key, time_string])
                        print branch
                        asset_grp = hdf.create_group(branch)
                        # Loop over the classifications to pull out the annotations for this asset.
                        all_x_pix = []
                        all_y_pix = []
                        all_user = []
                        # Counter for the number of submissions on this asset
                        submission_count = 0
                        # Iterate through classification rows to find annotations on this asset
                        for idc, cla in clas_subset.iterrows():
                            # Loop through values and find value with frame matching asset index
                            for val in cla['annotations']['value']:
                                # Does this frame match the asset of interest?
                                if val['frame'] == asset_id:
                                    # Get the data for this user classification
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [int(p['y']) for p in val['points']]
                                    user_id = cla['user_id']
                                    # Set up a group for this user.
                                    branch = "users/user_{}".format(submission_count)
                                    print "---->" + branch
                                    user_grp = asset_grp.create_group(branch)
                                    # Assign the user data to their group
                                    user_grp.create_dataset('x_pix', data=np.array(user_x_pix, dtype=int))
                                    user_grp.create_dataset('y_pix', data=np.array(user_y_pix, dtype=int))
                                    user_grp.create_dataset('id', data=user_id)
                                    # Dump users data into collection of annotations.
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    all_user.append(user_id)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        asset_grp.create_dataset('x_pix', data=np.array(all_x_pix, dtype=int))
                        asset_grp.create_dataset('y_pix', data=np.array(all_y_pix, dtype=int))
                        asset_grp.create_dataset('user_list', data=np.array(all_user, dtype=int))
    hdf.close()

def match_user_classifications_to_ssw_events(username, active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data structure containing linking each
    classification of a specific user with it's corresponding frame.
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
    name = "user_{}_classifications_matched_ssw_events.hdf5".format(username)
    out_hdf5_name = os.path.join(project_dirs['out_data'], name)
    if os.path.exists(out_hdf5_name):
        os.remove(out_hdf5_name)

    # Create the hdf5 output files
    hdf = h5py.File(out_hdf5_name, "w")

    for event_key, craft in event_tree.iteritems():

        for craft_key, im_type in craft.iteritems():

            for im_key, subjects in im_type.iteritems():

                for sub in subjects:
                    # Get the subjects for this subject_id
                    subjects_subset = get_df_subset(all_subjects, 'subject_id', [sub], get_range=False)
                    meta = subjects_subset['metadata'].values[0]
                    # Get the users classifications for this subset
                    clas_subset = get_df_subset(user_classifications, 'subject_id', [sub], get_range=False)
                    for asset_id in range(3):
                        asset_key = "asset_{}".format(asset_id)
                        # Form a timestring for the hdf5 branch name
                        file_parts = os.path.splitext(meta[asset_key])[0].split('_')
                        time_string = "_".join(file_parts[6:8])
                        time_string = pd.datetime.strptime(time_string, '%Y%m%d_%H%M%S').isoformat()
                        # Make a branch for this image type / asset time.
                        branch = "/".join([event_key, craft_key, im_key, time_string])
                        print branch
                        asset_grp = hdf.create_group(branch)
                        # Loop over the classifications to pull out the annotations for this asset.
                        all_x_pix = []
                        all_y_pix = []
                        # Counter for the number of submissions by this user on this asset
                        submission_count = 0
                        # Iterate through classification rows to find annotations on this asset
                        for idc, cla in clas_subset.iterrows():
                            # Loop through values and find value with frame matching asset index
                            for val in cla['annotations']['value']:
                                # Does this frame match the asset of interest?
                                if val['frame'] == asset_id:
                                    # Get the data for this user classification
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [int(p['y']) for p in val['points']]
                                    n_points = len(user_x_pix)
                                    # Set up a group for this user.
                                    branch = "classifications/classification_{}".format(submission_count)
                                    print "---->" + branch
                                    user_grp = asset_grp.create_group(branch)
                                    # Assign the user data to their group
                                    user_grp.create_dataset('x_pix', data=np.array(user_x_pix, dtype=int))
                                    user_grp.create_dataset('y_pix', data=np.array(user_y_pix, dtype=int))
                                    user_grp.create_dataset('n_points', data=n_points)
                                    # Dump users data into collection of annotations.
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        asset_grp.create_dataset('x_pix', data=np.array(all_x_pix, dtype=int))
                        asset_grp.create_dataset('y_pix', data=np.array(all_y_pix, dtype=int))
                        asset_grp.create_dataset('classification_count', data=np.array(submission_count, dtype=int))
    hdf.close()

def test_plot():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events.hdf5')

    hdf = h5py.File(out_hdf5_name, "r")
    for event in hdf.values():

        for craft_k, craft in event.iteritems():

            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            for i, (im_k, img) in enumerate(craft.iteritems()):

                for time_k, times in img.iteritems():

                    print craft_k, im_k, time_k



                        # Get time lims.
                        norm = mpl.colors.Normalize(vmin=0, vmax=len(times.keys()))
                        cmap = mpl.cm.viridis
                        c = 0
                        for k, v in times.iteritems():
                            a.plot(v['x_pix'][...], v['y_pix'][...], 'o', color=cmap(norm(c)))
                            a.set_title("Tracked in {0} {1} image".format(grp.attrs['craft'], im_type))
                            c += 1

                    for a in ax:
                        a.set_xlim(0, 1023)
                        a.set_ylim(0, 1023)
                        a.set_aspect('equal')

                    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
                    name = os.path.join(project_dirs['figs'], grp.attrs['craft'] + '_test.jpg')
                    plt.savefig(name)
                    plt.close('all')


def test_animation():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test animation.
    :return:
    """
    project_dirs = get_project_dirs()
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'out_data.hdf5')

    hdf = h5py.File(out_hdf5_name, "r")

    for grp in hdf.values():
        norm = grp['norm']
        diff = grp['diff']
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        im_count = 0
        # Get time lims.
        cm_norm = mpl.colors.Normalize(vmin=0, vmax=len(norm.keys()))
        cmap = mpl.cm.viridis
        c = 0
        craft = grp.attrs['craft']
        for norm_t, diff_t in zip(norm.itervalues(), diff.itervalues()):
            ax[0].plot(norm_t['x_pix'][...], norm_t['y_pix'][...], 'o', color=cmap(cm_norm(c)))
            ax[0].set_title("Tracked in {0} {1} image".format('craft', 'norm'))

            ax[1].plot(diff_t['x_pix'][...], diff_t['y_pix'][...], 'o', color=cmap(cm_norm(c)))
            ax[1].set_title("Tracked in {0} {1} image".format('craft', 'diff'))
            c += 1

            for a in ax:
                a.set_xlim(0, 1023)
                a.set_ylim(0, 1023)
                a.set_aspect('equal')

            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
            name = os.path.join(project_dirs['figs'], grp.attrs['craft']+"_ani_f{0:03d}.jpg".format(im_count))
            print name
            plt.savefig(name)
            im_count += 1
        plt.close('all')

    # Make animation and clean up.
    for craft in ['sta', 'stb']:
        src = os.path.join(project_dirs['figs'], craft + "*_ani*.jpg")
        dst = os.path.join(project_dirs['figs'], craft + "_front_id_test.gif")
        cmd = " ".join(["convert -delay 0 -loop 0 -resize 50%", src, dst])
        os.system(cmd)
        # Tidy.
        files = glob.glob(src)
        for f in files:
            os.remove(f)


def test_front_density():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test animation.
    :return:
    """
    project_dirs = get_project_dirs()
    out_hdf5_name = os.path.join(project_dirs['out_data'], 'out_data.hdf5')

    hdf = h5py.File(out_hdf5_name, "r")

    for grp in hdf.values():
        norm = grp['norm']
        diff = grp['diff']
        im_count = 0
        # Get time lims.
        cm_norm = mpl.colors.Normalize(vmin=0, vmax=len(norm.keys()))
        cmap = mpl.cm.viridis
        cmap.set_bad(color='k')
        c = 0
        craft = grp.attrs['craft']

        x = np.arange(0, 1023)
        y = np.arange(0, 1023)
        xm, ym = np.meshgrid(x, y)
        positions = np.vstack([xm.ravel(), ym.ravel()])

        for norm_t, diff_t in zip(norm.itervalues(),diff.itervalues()):

            plot_front = False
            if (len(norm_t['x_pix'])>5) & (len(diff_t['x_pix'])>5):
                plot_front = True
                values = np.vstack([norm_t['x_pix'], norm_t['y_pix']])
                norm_kernel = stats.gaussian_kde(values)

                values = np.vstack([diff_t['x_pix'], diff_t['y_pix']])
                diff_kernel = stats.gaussian_kde(values)

                norm_cme_loc = np.reshape(norm_kernel(positions).T, xm.shape)
                diff_cme_loc = np.reshape(diff_kernel(positions).T, xm.shape)

            ###########################################################################
            fig, ax = plt.subplots(1, 2, figsize=(14, 7.05))
            if plot_front:
                q = np.percentile(norm_cme_loc, [90, 92.5, 95, 97.5, 100])
                ax[0].contourf(xm, ym, norm_cme_loc, levels=q, origin='lower', cmap=cmap, alpha=0.5)
                q = np.percentile(diff_cme_loc, [90, 92.5, 95, 97.5, 100])
                ax[1].contourf(xm, ym, diff_cme_loc, levels=q, origin='lower', cmap=cmap, alpha=0.5)

            ax[0].plot(norm_t['x_pix'][...], norm_t['y_pix'][...], 'o', color='r')
            ax[0].set_title("Tracked in {0} {1} image".format('craft', 'norm'))

            ax[1].plot(diff_t['x_pix'][...], diff_t['y_pix'][...], 'o', color='r')
            ax[1].set_title("Tracked in {0} {1} image".format('craft', 'diff'))
            c += 1

            for a in ax:
                a.set_xlim(0, 1023)
                a.set_ylim(0, 1023)
                a.set_aspect('equal')

            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
            name = os.path.join(project_dirs['figs'], grp.attrs['craft']+"_ani_CME_density_f{0:03d}.jpg".format(im_count))
            print name
            plt.savefig(name)
            plt.close('all')
            im_count += 1


    # Make animation and clean up.
    for craft in ['sta', 'stb']:
        src = os.path.join(project_dirs['figs'], craft + "*_ani_CME_density*.jpg")
        dst = os.path.join(project_dirs['figs'], craft + "_front_density_test.gif")
        cmd = " ".join(["convert -delay 0 -loop 0 -resize 50%", src, dst])
        os.system(cmd)
        # Tidy.
        files = glob.glob(src)
        for f in files:
            os.remove(f)
