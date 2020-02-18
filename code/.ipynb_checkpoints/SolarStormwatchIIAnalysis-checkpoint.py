import glob
import os
import tables
import astropy.units as u
import heliospheric_imager_analysis.images as hip
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simplejson as json
from sklearn.neighbors import KernelDensity
from skimage import measure

def get_project_dirs():
    """
    A function to load in dictionary of project directories stored in a config.txt file stored in the
    :return proj_dirs: Dictionary of project directories, with keys 'data','figures','code', and 'results'.
    """
    files = glob.glob('C:\\Users\\yq904481\\research\\repos\\SolarStormwatchIIAnalysis\\code\\project_directories.txt')
    
    if len(files) != 1:
        # If too few or too many config files, guess projdirs
        print('Error: Cannot find correct config file with project directories. Assuming typical structure')
        proj_dirs = {'data': os.path.join(os.getcwd(), 'data'), 'figs': os.path.join(os.getcwd(), 'figures'),
                     'code': os.path.join(os.getcwd(), 'code'), 'results': os.path.join(os.getcwd(), 'results'),
                     'hi_data': os.path.join(os.getcwd(), r'STEREO\ares.nrl.navy.mil\lz\L2_1_25'),
                     'out_data': os.path.join(os.getcwd(), 'out_data'),
                     'swpc_data': os.path.join(os.getcwd(), 'data\\enlil_cme_analysis_single_events.xlsx'),
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
    for val in proj_dirs.values():
        if not os.path.exists(val):
            print('Error, invalid path, check config: ' + val)

    return proj_dirs


def load_swpc_events():
    """
    Function to load in the excel file containing the SWPC CMEs provided by Curt. Import as a pandas dataframe. There is
    no official source for this data.
    :return:
    """
    # TODO: Ask Curt for source and acknowledgement for this data?
    proj_dirs = get_project_dirs()
    # the swpc cmes has column names [event_id	t_appear	t_start	t_submission	lat	lon	half_width	vel	t_ace_obs
    # 	t_ace_wsa	diff	late_or_early	t_earth_si	Comment]. Get converters for making sure times ported correctly
    # Column 'diff' is awkward, as is a time delta, so handle seperately, by parsing to string first
    convert = dict(t_appear=pd.to_datetime, t_start=pd.to_datetime, t_submission=pd.to_datetime,
                   t_ace_obs=pd.to_datetime, t_ace_wsa=pd.to_datetime, t_earth_si=pd.to_datetime,
                   t_hi1a_start=pd.to_datetime, t_hi1a_stop=pd.to_datetime, t_hi1b_start=pd.to_datetime,
                   t_hi1b_stop=pd.to_datetime, diff=str)

    swpc_cmes = pd.read_excel(proj_dirs['swpc_data'], header=0, index_col=None, converters=convert)
    swpc_cmes['diff'] = pd.to_timedelta(swpc_cmes['diff'], unit='h')
    return swpc_cmes

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
    converters = dict(classification_id=int, user_name=str, user_ip=str, workflow_id=int,
                      workflow_name=str, workflow_version=float, created_at=str, gold_standard=str, expert=str,
                      metadata=json.loads, annotations=json.loads, subject_data=json.loads, subject_ids=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['classifications'], converters=converters)

    # Correct empty user_ids and convert to integers.
    data['user_id'].replace(to_replace=[np.NaN],value=-1, inplace=True)
    data['user_id'].astype(int)

    # Convert the subject ids, which are a variable length list of subject_ids. Because of setup of workflow
    # these numbers just repeat. Only need first.
    # Also, the json annotations are buried in a list, but only the dictionary in element 0 is needed. So pull this out
    # too
    subject_ids = []
    annos = []
    for idx, row in data.iterrows():
        subject_ids.append(list(row['subject_data'].keys())[0])
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
    converters = dict(subject_id=int, project_id=int, workflow_id=str, subject_set_id=int, metadata=json.loads,
                  locations=str, classifications_by_workflow=str, retired_in_workflow=str)
    # Load the classifications into DataFrame and get latest subset if needed.
    data = pd.read_csv(project_dirs['subjects'], converters=converters)

    id_bad = data.loc[:,'workflow_id']==''
    data.loc[id_bad, 'workflow_id'] = 'NaN'
    data['workflow_id'] = data['workflow_id'].astype(float)

    if active:
        id_active = np.logical_not(data['workflow_id'].isnull())
        data = data[id_active]

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
    
    *** HACKED TO AVOID ssw_code==ssw_000, as this doesn't have same form as other events for some reason ***
    """
    subjects = import_subjects(active=active)

    event_ids = []

    for ids, sub in subjects.iterrows():
        name = sub['metadata']['subject_name'].split('_')
        ssw_code = "_".join(name[0:2])
        if ssw_code != "ssw_000":
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
        if ssw_code != "ssw_000":
            craft = name[4]
            im_type = name[5]
            event_tree[ssw_code][craft][im_type].append(sub['subject_id'])

    return event_tree

def test_out_of_range():
    project_dirs = get_project_dirs()

    # Get classifications from latest workflow
    classifications = import_classifications(latest=True)
    x_lo = 0
    x_hi = 0
    y_lo = 0
    y_hi = 0
    tot_bad = 0
    bad_clas = []
    store_id = []
    for idc, cla in classifications.iterrows():
        # Loop through values and find value with frame matching asset index
        for val in cla['annotations']['value']:
                # Get the pixel coordinates
                x_pix = np.array([int(p['x']) for p in val['points']])
                y_pix = np.array([1023 - int(p['y']) for p in val['points']])
                if any((x_pix<0)):
                    x_lo += 1
                if any((x_pix>1024)):
                    x_hi += 1
                if any((y_pix<0)):
                    y_lo += 1
                if any((y_pix>1024)):
                    y_hi += 1
                if any((x_pix < 0)) | any((x_pix>1024)) | any((y_pix<0)) | any((y_pix>1024)):
                    tot_bad +=1
                    bad_clas.append([x_pix, y_pix])
                    store_id.append(idc)

    print("x_lo: {}".format(x_lo))
    print("x_hi: {}".format(x_hi))
    print("y_lo: {}".format(y_lo))
    print("y_hi: {}".format(y_hi))
    print("Total number of Classifications with bad points: {}".format(tot_bad))
    print("Total number of Classifications: {}".format(idc))
    i = 0
    while i < 10:
        print(store_id[i])
        print(bad_clas[i][0])
        print( bad_clas[i][1])
        print("***************************")
        i += 1

def match_all_classifications_to_ssw_events(active=True, latest=True):
    """
    A function to process the Solar Stormwatch subjects to create a HDF5 data structure linking each
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
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events_plus_HPR.hdf5')
    if os.path.exists(ssw_out_name):
        os.remove(ssw_out_name)

    # Create the hdf5 output file
    ssw_out = tables.open_file(ssw_out_name, mode="w", title="SolarStormwatch CME classifications for all users")

    # Define the table types needed:
    # Table definition for a users pixels and HPR coords
    class UserCoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        el = tables.Float32Col()  # 32-bit float
        pa = tables.Float32Col()  # 32-bit float

    # Table definition for all users pixels and HPR coords, with added weight, based on num of points in users profile.
    class AllCoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        weight = tables.Int32Col()  # 32-bit integer
        el = tables.Float32Col()  # 32-bit float
        pa = tables.Float32Col()  # 32-bit float

    # Table definition for CME Front in pixel and HPR coordinates.
    class CMECoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        x_lo = tables.Int32Col()  # 32-bit integer
        y_lo = tables.Int32Col()  # 32-bit integer
        x_hi = tables.Int32Col()  # 32-bit integer
        y_hi = tables.Int32Col()  # 32-bit integer
        pa = tables.Float32Col()  # 32-bit float
        el = tables.Float32Col()  # 32-bit float
        el_lo = tables.Float32Col()  # 32-bit float
        el_hi = tables.Float32Col()  # 32-bit float

    # Loop through the event tree, making relevant groups in the ssw_out file.
    # Group structure: Event> Craft> Img type> All classifications> User classifications
    for event_key, craft in event_tree.items():

        event_group = ssw_out.create_group("/", event_key, event_key)

        for craft_key, im_type in craft.items():

            craft_group = ssw_out.create_group(event_group, craft_key, craft_key)

            for im_key, subjects in im_type.items():

                im_group = ssw_out.create_group(craft_group, im_key, im_key)

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
                        if file_parts[7][2:4] in {'09', '29', '49'}:
                            hhmmss = file_parts[7][0:4] + '01'
                        else:
                            # Add in a correction for an error in the asset names that makes file times offset from
                            # the HI data source. All data file times end 0901, 4901 or 2901.
                            # Force to make it easier to look up files. This handles known cases, but maybe not all.
                            if file_parts[7][2:4] == '10':
                                hhmmss = file_parts[7][0:2] + '0901'
                            if file_parts[7][2:4] == '30':
                                hhmmss = file_parts[7][0:2] + '2901'
                            if file_parts[7][2:4] == '50':
                                hhmmss = file_parts[7][0:2] + '4901'

                        time_string = "T" + "_".join([file_parts[6], hhmmss])
                        label = "_".join([event_key,craft_key,im_key,time_string])
                        print(label)
                        asset_time = pd.datetime.strptime(time_string, 'T%Y%m%d_%H%M%S')

                        # Get the HI file and sunpy map for this asset
                        hi_files = hip.find_hi_files(asset_time, asset_time, craft=craft_key,
                                                     camera='hi1', background_type=1)
                        asset_hi_map = hip.get_image_plain(hi_files[0])

                        # Make a group for this asset
                        asset_group = ssw_out.create_group(im_group, time_string, title=asset_time.isoformat())
                        # Form lists for storing all of the user classifications, id numbers and weights
                        all_x_pix = []
                        all_y_pix = []
                        all_hpr_el = []
                        all_hpr_pa = []
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

                                    # Get the pixel coordinates
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [1023 - int(p['y']) for p in val['points']]
                                    # Convert to HPR
                                    user_el, user_pa = hip.convert_pix_to_hpr(user_x_pix*u.pix, user_y_pix*u.pix,
                                                                              asset_hi_map)
                                    # Loose the units and convert to float32
                                    user_el = np.array(user_el.value, dtype=np.float32)
                                    user_pa = np.array(user_pa.value, dtype=np.float32)

                                    # Make a tables for this users annotation and get pointer
                                    table = ssw_out.create_table(user, 'coords', UserCoords,
                                                                 expectedrows=len(user_x_pix),
                                                                 title="Pixel adn HPR Coords")
                                    coord = table.row
                                    # Append the data
                                    for x, y, el, pa in zip(user_x_pix, user_y_pix, user_el, user_pa):
                                        coord['x'] = x
                                        coord['y'] = y
                                        coord['el'] = el
                                        coord['pa'] = pa
                                        coord.append()
                                    # Write to file
                                    table.flush()

                                    # Dump users data into collection of annotations.
                                    # Get the data for this user classification
                                    user_weight = np.zeros(len(user_x_pix), dtype=int) + len(user_x_pix)
                                    user_id = cla['user_id']
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    all_hpr_el.extend(user_el.tolist())
                                    all_hpr_pa.extend(user_pa.tolist())
                                    all_user_weight.extend(user_weight)
                                    all_user.append(user_id)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        # Make table and get pointer
                        table = ssw_out.create_table(asset_group, 'coords', AllCoords,
                                                         expectedrows=len(all_x_pix), title="Pixel and HPR coords")
                        coord = table.row
                        for x, y, el, pa, w in zip(all_x_pix, all_y_pix, all_hpr_el, all_hpr_pa, all_user_weight):
                            coord['x'] = x
                            coord['y'] = y
                            coord['el'] = el
                            coord['pa'] = pa
                            coord['weight'] = w
                            coord.append()
                        # Write to file.
                        table.flush()

                        pix_coords = pd.DataFrame({'x': all_x_pix, 'y': all_y_pix})
                        hpr_coords = pd.DataFrame({'el': all_hpr_el, 'pa': all_hpr_pa})

                        # Now use kernal density estimation to identify pixel coordinates of the CME front.
                        cme_coords = kernel_estimate_cme_front(pix_coords, hpr_coords, asset_hi_map, kernel="epanechnikov", bandwidth=40,
                                                               thresh=10)

                        # Now store the CME front coords: Get table and pointer
                        table = ssw_out.create_table(asset_group, 'cme_coords', CMECoords,expectedrows=len(cme_coords),
                                                     title="Pixel and HPR coords of CME front")
                        coord = table.row
                        # Loop through cme coords and write to file.
                        for idr, cme_row in cme_coords.iterrows():
                            coord['x'] = np.int(cme_row['x'])
                            coord['y'] = np.int(cme_row['y'])
                            coord['x_lo'] = np.int(cme_row['x_lo'])
                            coord['y_lo'] = np.int(cme_row['y_lo'])
                            coord['x_hi'] = np.int(cme_row['x_hi'])
                            coord['y_hi'] = np.int(cme_row['y_hi'])
                            coord['pa'] = np.float32(cme_row['pa'])
                            coord['el'] = np.float32(cme_row['el'])
                            coord['el_lo'] = np.float32(cme_row['el_lo'])
                            coord['el_hi'] = np.float32(cme_row['el_hi'])
                            coord.append()
                        # Write to file.
                        table.flush()
    ssw_out.close()


def match_user_classifications_to_ssw_events(username, active=True, latest=True):
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

    # Create the hdf5 output file
    title = "SolarStormwatch CME classifications for {}".format(username)
    ssw_out = tables.open_file(ssw_out_name, mode="w", title=title)

    # Define the table types needed:
    # Table definition for a users pixels and HPR coords
    class UserCoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        el = tables.Float32Col()  # 32-bit float
        pa = tables.Float32Col()  # 32-bit float

    # Table definition for all users pixels and HPR coords, with added weight, based on num of points in users profile.
    class AllCoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        weight = tables.Int32Col()  # 32-bit integer
        el = tables.Float32Col()  # 32-bit float
        pa = tables.Float32Col()  # 32-bit float

    # Table definition for CME Front in pixel and HPR coordinates.
    class CMECoords(tables.IsDescription):
        x = tables.Int32Col()  # 32-bit integer
        y = tables.Int32Col()  # 32-bit integer
        x_lo = tables.Int32Col()  # 32-bit integer
        y_lo = tables.Int32Col()  # 32-bit integer
        x_hi = tables.Int32Col()  # 32-bit integer
        y_hi = tables.Int32Col()  # 32-bit integer
        el = tables.Float32Col()  # 32-bit float
        pa = tables.Float32Col()  # 32-bit float

    # Loop through the event tree, making relevant groups in the ssw_out file.
    # Group structure: Event> Craft> Img type> All classifications> User classifications
    for event_key, craft in event_tree.iteritems():
        event_group = ssw_out.create_group("/", event_key, event_key)
        print("Processing event {}".format(event_key))
        for craft_key, im_type in craft.iteritems():
            craft_group = ssw_out.create_group(event_group, craft_key, craft_key)

            for im_key, subjects in im_type.iteritems():
                im_group = ssw_out.create_group(craft_group, im_key, im_key)

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
                        if file_parts[7][2:4] in {'09', '29', '49'}:
                            hhmmss = file_parts[7][0:4] + '01'
                        else:
                            # Add in a correction for an error in the asset names that makes file times offset from
                            # the HI data source. All data file times end 0901, 4901 or 2901.
                            # Force to make it easier to look up files. This handles known cases, but maybe not all.
                            if file_parts[7][2:4] == '10':
                                hhmmss = file_parts[7][0:2] + '0901'
                            if file_parts[7][2:4] == '30':
                                hhmmss = file_parts[7][0:2] + '2901'
                            if file_parts[7][2:4] == '50':
                                hhmmss = file_parts[7][0:2] + '4901'

                        time_string = "T" + "_".join([file_parts[6], hhmmss])
                        asset_time = pd.datetime.strptime(time_string, 'T%Y%m%d_%H%M%S')

                        # Get the HI file and sunpy map for this asset
                        hi_files = hip.find_hi_files(asset_time, asset_time, craft=craft_key,
                                                     camera='hi1', background_type=1)
                        asset_hi_map = hip.get_image_plain(hi_files[0])

                        # Make a group for this asset
                        asset_group = ssw_out.create_group(im_group, time_string, title=asset_time.isoformat())
                        # Form lists for storing all of the user classifications, id numbers and weights
                        all_x_pix = []
                        all_y_pix = []
                        all_hpr_el = []
                        all_hpr_pa = []
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

                                    # Get the pixel coordinates
                                    user_x_pix = [int(p['x']) for p in val['points']]
                                    user_y_pix = [1023 - int(p['y']) for p in val['points']]
                                    # Convert to HPR
                                    user_el, user_pa = hip.convert_pix_to_hpr(user_x_pix*u.pix, user_y_pix*u.pix,
                                                                              asset_hi_map)
                                    # Loose the units and convert to float32
                                    user_el = np.array(user_el.value, dtype=np.float32)
                                    user_pa = np.array(user_pa.value, dtype=np.float32)

                                    # Make a tables for this users annotation and get pointer
                                    table = ssw_out.create_table(clas, 'coords', UserCoords,
                                                                 expectedrows=len(user_x_pix),
                                                                 title="Pixel adn HPR Coords")
                                    coord = table.row
                                    # Append the data
                                    for x, y, el, pa in zip(user_x_pix, user_y_pix, user_el, user_pa):
                                        coord['x'] = x
                                        coord['y'] = y
                                        coord['el'] = el
                                        coord['pa'] = pa
                                        coord.append()
                                    # Write to file
                                    table.flush()

                                    # Dump users data into collection of annotations.
                                    # Get the data for this user classification
                                    user_weight = np.zeros(len(user_x_pix), dtype=int) + len(user_x_pix)
                                    user_id = cla['user_id']
                                    all_x_pix.extend(user_x_pix)
                                    all_y_pix.extend(user_y_pix)
                                    all_hpr_el.extend(user_el.tolist())
                                    all_hpr_pa.extend(user_pa.tolist())
                                    all_user_weight.extend(user_weight)
                                    all_user.append(user_id)
                                    submission_count += 1

                        # Now add on the collected annotations to the assets group.
                        # Make table and get pointer
                        table = ssw_out.create_table(asset_group, 'coords', AllCoords,
                                                         expectedrows=len(all_x_pix), title="Pixel and HPR coords")
                        coord = table.row
                        for x, y, el, pa, w in zip(all_x_pix, all_y_pix, all_hpr_el, all_hpr_pa, all_user_weight):
                            coord['x'] = x
                            coord['y'] = y
                            coord['el'] = el
                            coord['pa'] = pa
                            coord['weight'] = w
                            coord.append()
                        # Write to file.
                        table.flush()

                        coords = pd.DataFrame({'x': all_x_pix, 'y': all_y_pix})
                        # Now use kernal density estimation to identify pixel coordinates of the CME front.
                        cme_coords = kernel_estimate_cme_front(coords, kernel="epanechnikov", bandwidth=40,
                                                               thresh=10)
                        if len(cme_coords)>0:
                            # Also get the HPR coords of CME front.
                            el, pa = hip.convert_pix_to_hpr(cme_coords['x'].values*u.pix, cme_coords['y'].values*u.pix,
                                                            asset_hi_map)
                            cme_coords['el'] = el.value
                            cme_coords['pa'] = pa.value
                        else:
                            # Set invalid values
                            cme_coords = pd.DataFrame({'x': 99999, 'y': 99999, 'el': 99999, 'pa': 99999}, index=[0])

                        # Now store the CME front coords: Get table and pointer
                        table = ssw_out.create_table(asset_group, 'cme_coords', CMECoords,expectedrows=len(cme_coords),
                                                     title="Pixel and HPR coords of CME front")
                        coord = table.row
                        # Loop through cme coords and write to file.
                        for idr, cme_row in cme_coords.iterrows():
                            coord['x'] = np.int(cme_row['x'])
                            coord['y'] = np.int(cme_row['y'])
                            coord['el'] = np.float32(cme_row['el'])
                            coord['pa'] = np.float32(cme_row['pa'])
                            coord.append()
                        # Write to file.
                        table.flush()
    ssw_out.close()


def get_fwhm(x, pdf):
    """
    Function to calculate the full width at half maximum of a 1-d distribution.
    :param x:
    :param pdf:
    :return:
    """

    #Calc FWHM
    pdf_max = pdf.max()
    pdf_hm = pdf_max/2.0
    id_max = np.argwhere(pdf==pdf_max).ravel()
    if len(id_max)>1:
        id_max = id_max[0]
    x_max = x[id_max]

    # Look low.
    pdf_l = pdf[x<x_max]
    x_l = x[x<x_max]
    if len(x_l)>1:
        x_hm_l = x_l[np.argmin(np.abs(pdf_l - pdf_hm))]
    else:
        x_hm_l = np.NaN

    # Look high.
    pdf_h = pdf[x>x_max]
    x_h = x[x>x_max]
    if len(x_h)>1:
        x_hm_h = x_h[np.argmin(np.abs(pdf_h - pdf_hm))]
    else:
        x_hm_h = np.NaN

    return x_hm_l, x_max, x_hm_h


def kernel_estimate_cme_front(pix_coords, hpr_coords, hi_map, kernel="epanechnikov", bandwidth=40, thresh=10):
    """
    A function that uses Kernel density estimation and skeletonization to estimate the CME front location from the
    cloud of classifications for this asset.
    :param pix_coords: Dataframe of classification coordinates, with columns 'x', and 'y' for the x pixels and y pixels.
    :param hpr_coords: Dataframe of classification coordinates, with columns 'pa', and 'el' for the pixel coords in HPR.
    :param hi_map: A sunpy map of the HI image current being analysed.
    :param kernel: String name of any valid kernal in scikit-learn.neighbours.KernalDensity.
    :param bandwidth: Float value of the bandwidth to be used by the kernal. Defaults to 40 pixels, as this is
                      approximately the error in classifications if made using a touchscreen.
    :param thresh: Float value of threshold to apply to the density map to identify the CME skeleton. Defaults to 10,
                   but this is only from playing around and hasn't been thoroughly testted.
    :return: coords: A dataframe with columns:
                     x: x-pixel coords of the best estimate of the cme front
                     y: y-pixel coords of the best estimate of the cme front
                     pa:
                     el:
                     el_lo:
                     el_hi:
    """
    if pix_coords.shape[0] < 20:
        print("Error: <20 coordinates provided, CME front identification likely to be poor")

    if kernel not in {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}:
        print("Error: invalid kernel, defaulting to epanechnikov")
        kernel = "epanechnikov"

    if not isinstance(bandwidth, (int, float)):
        print("Error: bandwidth should be an int or float number. defaulting to 40")
        bandwidth = 40

    if not isinstance(thresh, (int, float)):
        print("Error: thresh should be an int or float number. defaulting to 10")
        bandwidth = 10

    # Get all pixel coordinates in the image frame.
    x = np.arange(0, 1024)
    y = np.arange(0, 1024)
    xm, ym = np.meshgrid(x, y)
    # Get all coords for KSdensity
    all_coords = np.vstack([xm.ravel(), ym.ravel()]).T
    # Lookup HPR coords in frame
    elm, pam = hip.convert_pix_to_hpr(xm * u.pix, ym * u.pix, hi_map)
    # Loose the units as not needed
    elm = elm.value
    pam = pam.value

    # Parse classification cloud to kernel density estimator. Returns log of density.
    xy = np.vstack([pix_coords['x'].values.ravel(), pix_coords['y'].values.ravel()]).T
    # Get flags for aborting the CME fit distributions
    fit_distribution = True
    fit_cme_distribution = True

    if xy.shape[0] == 0:
        fit_distribution = False

    if fit_distribution:
        # Use first stab at kernel density to
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(xy)
        log_pdf = np.reshape(kde.score_samples(all_coords), xm.shape)
        # Convert the pdf to a binary map of where the distribution is > np.max(dist)/thresh
        pdf = np.exp(log_pdf.copy())
        thresh = np.max(pdf) / thresh
        pdf_binary = pdf.copy()
        pdf_binary[pdf_binary < thresh] = 0
        pdf_binary[pdf_binary >= thresh] = 1
        # Only keep the largest CME feature in the binary map.
        # Label each binary blob
        pdf_binary_label = measure.label(pdf_binary.astype(int))
        # update label so that the background pixels have the same label of zero
        pdf_binary_label[pdf_binary == 0] = 0

        # Find how many unique labels and how many pixels in each label
        lab, cnt = np.unique(pdf_binary_label, return_counts=True)
        # Sort these so largest is last in list.
        # Largest should always be the background. So keep two largest - background and clearest CME identification
        lab = lab[np.argsort(cnt)]
        for l in lab[:-2]:
            pdf_binary_label[pdf_binary_label == l] = 0

        # Set the largest blob back to 1 rather than it's label.
        pdf_binary_label[pdf_binary_label != 0] = 1

        # Find classifications inside this polygon
        id_keep = []
        for idr, row in pix_coords.iterrows():
            good_row = True
            if (row['y'] < 0) | (row['y'] > 1023):
                print("row: {0}, y : {1}".format(idr, row['y']))
                good_row = False
            if (row['x'] < 0) | (row['x'] > 1023):
                print("row: {0}, x : {1}".format(idr, row['x']))
                good_row = False

            if good_row:
                if pdf_binary_label[int(row['y']), int(row['x'])] == 1:
                    id_keep.append(True)
                else:
                    id_keep.append(False)
            else:
                id_keep.append(False)

        xy_in_cme = np.vstack([pix_coords['x'][id_keep].values.ravel(), pix_coords['y'][id_keep].values.ravel()]).T
        hpr_in_cme = np.vstack([hpr_coords['pa'][id_keep].values.ravel(), hpr_coords['el'][id_keep].values.ravel()]).T

        # Now fit only these points
        # Are there any points left?
        if xy_in_cme.shape[0] == 0:
            fit_cme_distribution = False

        if fit_cme_distribution:
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(xy_in_cme)
            log_pdf = np.reshape(kde.score_samples(all_coords), xm.shape)
            cme_pdf = np.exp(log_pdf.copy())

            # Use classifications to find range of PA values to look up CME elongation for.
            # Preallocate space for CME front.
            # Use ceil and floor so that pa_vals definitely stays in range.
            pa_vals = np.arange(np.ceil(hpr_in_cme[:,0].min()), np.floor(hpr_in_cme[:,0].max()), 1)
            cme_el_peak = np.zeros(pa_vals.shape)
            cme_el_lo = np.zeros(pa_vals.shape)
            cme_el_hi = np.zeros(pa_vals.shape)
            # Loop over position angles, get FWHM of the kernel density estimate along constant PA.
            # Use FWHM as elon error est.
            for i, pa in enumerate(pa_vals):
                # Get pixel coords of this PA slice
                pa_slice = measure.find_contours(pam,pa)[0]
                # Get values rounded to nearest pixel
                pa_slice = pa_slice.astype(int)
                pdf_pa = cme_pdf[pa_slice[:, 0], pa_slice[:, 1]]
                el_pa = elm[pa_slice[:, 0], pa_slice[:, 1]]
                cme_el_lo[i], cme_el_peak[i], cme_el_hi[i] = get_fwhm(el_pa, pdf_pa)

            if len(cme_el_peak) != 0:
                x, y = hip.convert_hpr_to_pix(cme_el_peak * u.deg, pa_vals * u.deg, hi_map)
                x_lo, y_lo = hip.convert_hpr_to_pix(cme_el_lo * u.deg, pa_vals * u.deg, hi_map)
                x_hi, y_hi = hip.convert_hpr_to_pix(cme_el_hi * u.deg, pa_vals * u.deg, hi_map)
                # Setup output dataframe
                cme_coords = pd.DataFrame({'x': x.value, 'y': y.value,
                                           'x_lo': x_lo.value, 'y_lo': y_lo.value,
                                           'x_hi': x_hi.value, 'y_hi': y_hi.value,
                                           'pa': pa_vals, 'el': cme_el_peak,
                                           'el_lo': cme_el_lo, 'el_hi': cme_el_hi})

                # Check any NaNs are set to bad val
                cme_coords[cme_coords.isnull()] = 99999
            else:
                fit_cme_distribution = False

    # Handle bad values case
    if (not fit_distribution) or (not fit_cme_distribution):
        # Set bad values to 99999, as pytables can't handle NaN
        bad_val = 99999
        keys = ['x', 'y', 'x_lo', 'y_lo', 'x_hi', 'y_hi', 'pa', 'el', 'el_lo', 'el_hi']
        bad_val_dict = {k:bad_val for k in keys}
        cme_coords = pd.DataFrame(bad_val_dict, index=[0])

    return cme_coords


def test_plot():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """
    project_dirs = get_project_dirs()
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events_plus_HPR.hdf5')

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

                    try:
                        all_coords = pd.DataFrame.from_records(t.coords.read())

                        cme_coords = pd.DataFrame.from_records(t.cme_coords.read())
                        cme_coords.replace(to_replace=[99999], value=np.NaN, inplace=True)
                        ax[i].plot(cme_coords['x'], cme_coords['y'], '-', color=cmap(norm(frame_count)), linewidth=2)
                        ax[i].plot(all_coords['x'], all_coords['y'], 'o', color=cmap(norm(frame_count)), alpha=0.5)
                    except:
                        print("No coords to plot")

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
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events_plus_HPR.hdf5')

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
                norm_all = pd.DataFrame.from_records(norm_t.coords.read())
                norm_cme = pd.DataFrame.from_records(norm_t.cme_coords.read())
                norm_cme.replace(to_replace=[99999], value=np.NaN, inplace=True)

                ax[0].plot(norm_cme['x'], norm_cme['y'], '-', color=cmap(norm(frame_count)), linewidth=2)
                ax[0].plot(norm_all['x'], norm_all['y'], 'o', color=cmap(norm(frame_count)), alpha=0.5)

                diff_all = pd.DataFrame.from_records(diff_t.coords.read())
                diff_cme = pd.DataFrame.from_records(diff_t.cme_coords.read())
                diff_cme.replace(to_replace=[99999], value=np.NaN, inplace=True)

                ax[1].plot(diff_cme['x'], diff_cme['y'], '-', color=cmap(norm(frame_count)))
                ax[1].plot(diff_all['x'], diff_all['y'], 'o', color=cmap(norm(frame_count)), alpha=0.5)

                # Label it up and format
                for a, img_k in zip(ax, ["norm", "diff"]):
                    a.set_title("Tracked in {0} {1} image".format(craft_k, img_k))
                    a.set_xlim(0, 1023)
                    a.set_ylim(0, 1023)
                    a.set_aspect('equal')

                # Save and move to next plot
                plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.92, wspace=0.075)
                name = "_".join([event_k,  craft_k, "ani_f{0:03d}.jpg".format(frame_count)])
                name = os.path.join(project_dirs['figs'], name)
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


def test_front_reconstruction():
    """
    A function to load in the test version of the HDF5 stormwatch data and produce a test plot.
    :return:
    """

    project_dirs = get_project_dirs()
    # Get the swpc cme database
    swpc_cmes = load_swpc_events()

    # Import the SSW classifications data
    ssw_out_name = os.path.join(project_dirs['out_data'], 'all_classifications_matched_ssw_events_plus_HPR.hdf5')
    ssw_out = tables.open_file(ssw_out_name, mode="r")

    # Make a directory to store all of these figures in
    fig_out_dir = os.path.join(project_dirs['figs'], "cme_front_reconstructions")
    if not os.path.exists(fig_out_dir):
        os.mkdir(fig_out_dir)

    # Iterate through the events, and annotate storm position on each plot.
    for event in ssw_out.iter_nodes('/'):

        # Get ssw event number to loop up event HI start and stop time.
        ssw_num = int(event._v_name.split('_')[1])

        for craft in event:
            print(craft._v_name)

            if craft._v_name == 'sta':
                t_start = swpc_cmes.loc[ssw_num, 't_hi1a_start']
                t_stop = swpc_cmes.loc[ssw_num, 't_hi1a_stop']
            elif craft._v_name == 'stb':
                t_start = swpc_cmes.loc[ssw_num, 't_hi1b_start']
                t_stop = swpc_cmes.loc[ssw_num, 't_hi1b_stop']

            hi_files = hip.find_hi_files(t_start, t_stop, craft=craft._v_name, camera='hi1', background_type=1)

            files_c = hi_files[1:]
            files_p = hi_files[0:-1]

            for img_type in craft:

                for fc, fp in zip(files_c, files_p):

                    if img_type._v_name == 'norm':
                        # Get Sunpy map of the image, convert to grayscale image with plain_normalise
                        hi_map = hip.get_image_plain(fc, star_suppress=False)
                        normalise = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
                        img = mpl.cm.gray(normalise(hi_map.data), bytes=True)
                    elif img_type._v_name == 'diff':
                        hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
                        normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
                        img = mpl.cm.gray(normalise(hi_map.data), bytes=True)

                    fig,ax = plt.subplots()
                    plt.imshow(img, origin='lower')

                    fc_base = os.path.basename(fc)
                    time_label = "T"+ "_".join(fc_base.split('_')[0:2])
                    if img_type.__contains__(time_label):
                        # Get the time node for this file.
                        time = img_type._f_get_child(time_label)
                        all = pd.DataFrame.from_records(time.coords.read())
                        all.replace(to_replace=[99999], value=np.NaN, inplace=True)
                        cme = pd.DataFrame.from_records(time.cme_coords.read())
                        cme.replace(to_replace=[99999], value=np.NaN, inplace=True)

                        # Add on all classifications
                        plt.plot(all['x'], all['y'], 'o', color='y', alpha=0.5)
                        # Add on CME front and error
                        plt.plot(cme['x'], cme['y'], '-', color='r', linewidth=2 )
                        plt.plot(cme['x_lo'], cme['y_lo'], '--', color='r', linewidth=2)
                        plt.plot(cme['x_hi'], cme['y_hi'], '--', color='r', linewidth=2)

                    else:
                        print('Error in retrieving points')
                        print(time_label)
                        print(img_type._v_children.keys())

                    plt.xlim(0, 1023)
                    plt.ylim(0, 1023)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    out_name = "_".join([event._v_name, craft._v_name, img_type._v_name,
                                         hi_map.date.strftime('%Y%m%d_%H%M%S'), 'plus_CME']) + '.jpg'
                    out_path = os.path.join(fig_out_dir, out_name)
                    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                    plt.close('all')

                src = os.path.join(fig_out_dir, '*plus_CME.jpg')
                gif_name = "_".join([event._v_name, craft._v_name, img_type._v_name, 'front_id_test.gif'])
                dst = os.path.join(fig_out_dir, gif_name)
                cmd = " ".join(["convert -delay 20 -loop 0 ", src, dst])
                os.system(cmd)
                # Tidy.
                files = glob.glob(src)
                for f in files:
                    os.remove(f)