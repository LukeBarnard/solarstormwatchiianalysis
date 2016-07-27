import glob
import os
import asset_production_tools as apt
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import scipy.ndimage as ndimage
import scipy.signal as signal
import sunpy.map as smap
import sunpy.image.coalignment as coalign


def find_hi_files(t_start, t_stop, craft="sta", camera="hi1", background_type=1):
    """
    Function to find a subset of the STEREO Heliospheric imager data.
    :param t_start: Datetime giving start time of data window requested
    :param t_stop: Datetime giving stop time of data window requested
    :param craft: String ['sta', 'stb'] to select data from either STEREO-A or STEREO-B.
    :param camera: String ['hi1', 'hi2'] to select data from either HI1 or HI2.
    :param background_type:  Integer [1, 11] to decide between selecting one or eleven day background subtraction.
    :return:
    """
    # STEREO HI data is stored on a directory tree with the following format:
    # level > background_type > craft > img > camera > daily_directories > hi_data_files

    # Get HI dirs.
    proj_dirs = apt.project_info()

    # Check the input arguments:
    if craft not in {'sta', 'stb'}:
        print("Error: camera should be set to either 'sta', or 'stb'. Defaulting to 'stb'")
        camera = 'sta'

    if camera not in {'hi1', 'hi2'}:
        print("Error: camera should be set to either 'hi1', or 'hi2'. Defaulting to 'hi1'")
        camera = 'hi1'

    if not isinstance(background_type, int):
        print("Error: background_type should be an integer, either 1 or 11. Defaulting to 1")
        background_type = 1

    if background_type not in {1, 11}:
        print("Error: background_type is invalid. Should be either 1, or 11. Defaulting to 1")
        background_type = 1

    # Work out the right directory names to get to the right part of data tree
    background_tag = "L2_" + str(background_type) + "_25"
    # Get path up to craft
    if craft == 'sta':
        craft_tag = 'a\img'
    elif craft == 'stb':
        craft_tag = 'b\img'

    # Get path up to craft
    if camera == 'hi1':
        camera_tag = 'hi_1'
    elif camera == 'hi2':
        camera_tag = 'hi_2'

    hi_path = os.path.join(proj_dirs['hi_data'], background_tag, craft_tag, camera_tag)

    # Use t_start/stop to get list of days to get data
    day_list = [t.strftime('%Y%m%d') for t in pd.date_range(t_start.date(), t_stop.date(), freq='1D')]

    all_files = []
    for day in day_list:
        path = os.path.join(hi_path, day + '\\*.fts')
        all_files.extend(glob.glob(path))

    # Out_files contains all files on dates corresponding to t_start/stop. Now restrict to the exact time window.
    t_min = t_start.strftime('%Y%m%d_%H%M%S')
    t_max = t_stop.strftime('%Y%m%d_%H%M%S')
    out_files = []
    for file_path in all_files:
        # Get the filename without full path
        file_name = os.path.basename(file_path)
        # HI files follow naming convention of yyyymmdd_hhmmss_datatag.fts. So first 15 elements give a time string.
        # TODO: Better way to pull out the timestring?
        time_tag = file_name[:15]
        # TODO:  ERROR?
        if (time_tag >= t_min) and (time_tag <= t_max):
            out_files.append(file_path)

    return out_files


def suppress_starfield(hi_map, thresh=97.5, res=512):
    """
    Function to suppress bright stars in the HI field of view. Is purely data based and does not use star-maps. Looks
    for large (high gradient) peaks by calculating the Laplacian of the image. Then uses morphological closing to
    identify the bright "tops" of stars, inside the high-gradient region. Then uses cubic interpolation
    (scipy.interp.bisplrep) to fill in pixels identified as a star. This is done in blocks over the image. This has
    only been developed with HI1 data - unsure how it will behave with HI2.
    :param hi_map: A sunpy map of the HI image the suppress the star field in.
    :param thresh: Float value containing the percentile threshold used to identify the large gradients associated with
                   stars. This means valid thresh values must lie in range 0-100, and should normally be high. e.g. 97.5
    :param res: Int value of block size (in pixels) to iterate over the image in.
    :return out_img: Star suppressed HI image.
    """
    # Check inputs
    if not isinstance(thresh, (float, int)):
        print("Error: Invalid thresh, should be float or int. Defaulting to 97.5")
        thresh = 97.5
    elif (thresh < 0) or (thresh > 100):
        print("Error: thresh = {} is invalid, should be in range 0-100. Defaulting to 97.5".format(thresh))
        thresh = 97.5

    if not isinstance(res, int):
        print("Error: Invalid res, should be an int. Defaulting to 512")
        res = 512
    elif (res < 0) or np.any((hi_map.data.shape < res)):
        print("Error: Invalid res, must be greater than zero and less than any of data dimensions")

    img = hi_map.data.copy()
    # Get del2 of image, to find horrendous gradients
    del2 = np.abs(ndimage.filters.laplace(img))
    # Find threshold of data, excluding NaNs
    thresh2 = np.percentile(del2[np.isfinite(del2)], thresh)
    abv_thresh = del2 > thresh2
    # Use binary closing to fill in big stars
    # TODO: Now fixed del2, can we remove the binary closing?
    # abv_thresh = ndimage.binary_closing(abv_thresh, structure=np.ones((3, 3)))

    if np.any(abv_thresh):
        star_r, star_c = np.nonzero(abv_thresh)
        good_vals = np.isfinite(img)
        nostar_r, nostar_c = np.nonzero(np.logical_and(~abv_thresh, good_vals))
    else:
        print('No points above threshold')
        out_img = img.copy()
        return out_img

    # Get interpolation block sizes.
    dr = res
    drp = 10
    dc = res
    dcp = 10
    out_img = img.copy()
    edge_pad = 5
    for r in range(0, img.shape[0], dr):

        if r == 0:
            # Add 5 pixel window at edge
            row_id_stars = np.logical_and(star_r >= (r + edge_pad), star_r <= (r + dr))
            row_id_nostars = np.logical_and(nostar_r >= (r - drp), nostar_r <= (r + dr + drp))
        elif 0 < r < (img.shape[0] - dr):
            row_id_stars = np.logical_and(star_r >= r, star_r <= (r + dr))
            row_id_nostars = np.logical_and(nostar_r >= (r - drp), nostar_r <= (r + dr + drp))
        elif r == (img.shape[0] - dr):
            # Add 5 pixel window at edge
            row_id_stars = np.logical_and(star_r >= r, star_r <= (r + dr - edge_pad))
            row_id_nostars = np.logical_and(nostar_r >= (r - drp), nostar_r <= (r + dr + drp))

        for c in range(0, img.shape[1], dr):

            if c == 0:
                # Add 5 pixel window at edge
                col_id_stars = np.logical_and(star_c > (c + edge_pad), star_c < (c + dc))
                col_id_nostars = np.logical_and(nostar_c > (c - dcp), nostar_c < (c + dc + dcp))
            elif 0 < c < (img.shape[1] - dc):
                col_id_stars = np.logical_and(star_c > c, star_c < (c + dc))
                col_id_nostars = np.logical_and(nostar_c > (c - dcp), nostar_c < (c + dc + dcp))
            elif c == (img.shape[1] - dc):
                # Add 5 pixel window at edge
                col_id_stars = np.logical_and(star_c > c, star_c < (c + dc - edge_pad))
                col_id_nostars = np.logical_and(nostar_c > (c - dcp), nostar_c < (c + dc + dcp))

            # Interpolate the padded image region.
            id_find = np.logical_and(row_id_nostars, col_id_nostars)
            x = nostar_c[id_find]
            y = nostar_r[id_find]
            f = interp.bisplrep(x, y, img[y, x], kx=3, ky=3)
            id_find = np.logical_and(row_id_stars, col_id_stars)
            x = star_c[id_find]
            y = star_r[id_find]
            for i, j in zip(y, x):
                out_img[i, j] = interp.bisplev(x, y, f)

    # TODO: Make a plot demonstrating how the star suppression works.
    hi_map.data = out_img.copy()
    return hi_map


def get_approx_star_field(img):
    """This function returns a binary array that provides a rough estimate of the locations of stars in the HI1 fov.
     All points above a fixed threshold are 1s, all points below are 0s. Used in the align_image, which is based
    on template matching against the background star-field.
    :param img: A HI image array
    :return img_stars: A binary image showing estimated locations of stars.
    """
    img_stars = img.copy()
    img_stars[~np.isfinite(img_stars)] = 0
    img_stars[img_stars < np.percentile(img_stars, 97.5)] = 0
    img_stars[img_stars != 0] = 1
    return img_stars


def align_image(src_map, dst_map):
    """
    Function to align two hi images. src_map is shifted by interpolation into the coordinates of dst_map. The
    transformation required to do this is calculated by pattern matching an approximation of the star field between
    frames in a subset of the HI image.
    :param src_map: A SunPy Map of the HI image to shift the coordinates of
    :param dst_map: A SunPy Map of the HI image to match coordinates against
    :return out_img: Array of src_map image shifted into coordinates of dst_map
    """
    # Note, this doesn't correctly update the header/meta information of src_map.
    mc = smap.MapCube([src_map, dst_map])
    # Calcualte the shifts needed to align the images, using sunpy.image.colaignment module.
    shifts = coalign.calculate_match_template_shift(mc, layer_index=1, func=get_approx_star_field)
    xshift = (shifts['x'].to('deg') / mc[0].scale.x)
    yshift = (shifts['y'].to('deg') / mc[0].scale.y)
    to_shift = [-yshift[0].value, -xshift[0].value]
    # TODO: Add in warning if shift is larger then some sensible value?
    # Deal with bad values in the image. Set to a the image median, keep record of the bad values.
    # Also shift the bad values, to mask out bad values in the shifted image. This is needed as shift routine
    # can't handle NaNs
    src_img = src_map.data.copy()
    # TODO: This method can probably be improved upon. Talk with Chris about this.
    img_avg = np.nanmedian(src_map.data)
    id_bad = np.isnan(src_map.data)
    src_img[id_bad] = img_avg
    # Now shift src_img and bad val mask.
    src_img_shft = ndimage.interpolation.shift(src_img, to_shift, mode='constant', cval=np.NaN)
    # TODO: Would it be better to lower the order on the mask interpolation? Atm, default order=3. Perhaps 1 or 0 more
    # TODO: approptiate for the mask interpolation?
    id_bad_shft = ndimage.interpolation.shift(id_bad.astype(float), to_shift, mode='constant', cval=1)
    # Correct bad_shft, round values to bad or good, convert to bool, set bad vals in image to nan.
    id_bad_shft = np.round(id_bad_shft).astype(bool)
    src_img_shft[id_bad_shft] = np.NaN
    src_map.data = src_img_shft.copy()
    return src_map


def get_image_plain(hi_file, star_suppress=False):
    """
    A function to load in a HI image file and return this as a SunPy Map object. Will optionally suppress the star field
    using hi_processing.filter_stars().
    :param hi_file: String, full path to a HI image file (in fits format).
    :param star_suppress: Bool, True or False on whether star suppression should be performed. Default false
    :return:
    """
    # Check inputs.
    if not os.path.exists(hi_file):
        print("Error: Path to file does not exist.")

    if not isinstance(star_suppress, bool):
        print("Error: star_suppress should be True or False. Defaulting to False")
        star_suppress = False

    hi_map = smap.Map(hi_file)

    if star_suppress:
        hi_map = suppress_starfield(hi_map)

    return hi_map


def get_image_diff(file_c, file_p, star_suppress=False, align=True, smoothing=False):
    """
    Function to produce a differenced image from HI data. Differenced image is calculated as Ic - Ip,
    loaded from file_c and file_p, respectively. Will optionally perform star field suppression (via
    hi_processing.suppress_starfield), and also image alignment (via hi_processing.align_image). Is currently
    only configured to do differences of consecutive images. Will return a blank frame if images
    given by file_c and file_p are separated by more then the nominal image cadence for hi1 or hi2, or come from
    different detectors.
    :param file_c: String, full path to file of image c.
    :param file_p: String, full path to file of image p.
    :param star_suppress: Bool, True or False on whether star suppression should be performed. Default False
    :param align: Bool, True or False depending on whether images should be aligned before differencing
    :param smoothing: Bool, True or False depending on whether the differenced image should by smoothed with a median
                      filter (5x5)
    :return:
    """
    if not os.path.exists(file_c):
        print("Error: Invalid path to file_c.")

    if not os.path.exists(file_p):
        print("Error: Invalid path to file_p.")

    if not isinstance(star_suppress, bool):
        print("Error: star_suppress should be True or False. Defaulting to False")
        star_suppress = False

    if not isinstance(align, bool):
        print("Error: align should be True or False. Defaulting to False")
        star_suppress = True

    if not isinstance(smoothing, bool):
        print("Error: align should be True or False. Defaulting to False")
        smoothing = True

    hi_c = smap.Map(file_c)

    hi_p = smap.Map(file_p)

    # Set flag to produce diff images, unless data checks fail.
    produce_diff_flag = True

    # Check data from same instrument
    if hi_c.nickname != hi_p.nickname:
        print("Error: Trying to differnece images from {0} and {1}.".format(hi_c.nickname, hi_p.nickname))
        produce_diff_flag = False

    # Check the images are only 1 image apart.
    if hi_c.detector == "HI1":
        # Get typical cadence of HI1 images
        cadence = pd.Timedelta(minutes=40)
        cadence_tol = pd.Timedelta(minutes=5)
    elif hi_c.detector == "HI2":
        # Get typical cadence of HI2 images
        cadence = pd.Timedelta(minutes=120)
        cadence_tol = pd.Timedelta(minutes=5)

    img_dt = hi_c.date - hi_p.date

    if np.abs((img_dt - cadence)) > cadence_tol:
        print("Error: Differenced images time difference is {0}, while typical cadence is {1}.".format(img_dt, cadence))
        print(" Returning a blank frame")
        produce_diff_flag = False

    if produce_diff_flag:
        # Align image p with image c,
        hi_p = align_image(hi_p, hi_c)

        if star_suppress:
            hi_c = suppress_starfield(hi_c)
            hi_p = suppress_starfield(hi_p)

        # Get difference image,
        hi_c.data = hi_c.data - hi_p.data

        # Apply some median smoothing.
        if smoothing:
            hi_c.data = signal.medfilt2d(hi_c.data, (5,5))
    else:
        hi_c.data = hi_c.data*np.NaN

    return hi_c