import numpy as np
import os
import sys
import imageio
import skimage.transform

from llff.poses.colmap_wrapper import run_colmap
import llff.poses.colmap_read_model as read_model

### THESE WERE ADDED HERE FOR FORMATTING FROM TXT FILES GENERATED BY HLOC NOTEBOOK ###
def qvec2rotmat_newformat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def format_cam_data(cam_dir):
    cam_array = []
    count = 1
    with open(cam_dir) as cam_file:
        for cam in cam_file:
            if count > 3:
                cam_array_parts = cam.split(" ")
                current_dict = {}
                current_dict["id"], current_dict["model"], current_dict["width"], current_dict["height"], current_dict["params"] = int(cam_array_parts[0]), cam_array_parts[1], int(cam_array_parts[2]),int(cam_array_parts[3]), cam_array_parts[4:]
                current_dict["params"][2] = current_dict["params"][2][:-1]
                current_dict["params"] = [float(k) for k in current_dict["params"]]
                cam_array.append(current_dict)
            count+=1
    return cam_array

def format_point_data(point_dir):
    point_array = []
    count = 1
    with open(point_dir) as point_file:
        for point in point_file:
            if count > 3:
                point_array_parts = point.split(" ")
                current_dict = {}
                current_dict["id"], current_dict["xyz"], current_dict["rgb"], current_dict["error"], current_dict["image_ids"], current_dict["point2D_idxs"] = int(point_array_parts[0]), [float(k) for k in point_array_parts[1:4]], [int(k) for k in point_array_parts[4:7]], float(point_array_parts[7]), [int(k) for k in point_array_parts[8::2]], point_array_parts[9::2]
                current_dict["point2D_idxs"][-1] = current_dict["point2D_idxs"][-1][:-1]
                current_dict["point2D_idxs"] = [int(k) for k in current_dict["point2D_idxs"]]
                point_array.append(current_dict)
            count+=1
    return point_array

def format_image_data(image_dir):
    image_array = []
    count = 1
    with open(image_dir) as image_file:
        while True:
            line1 = image_file.readline()
            line2 = image_file.readline()
            if not line2: break  # EOF
            if count > 4:
                image_array_parts = line1.split(" ")
                current_dict = {}
                current_dict["id"], current_dict["qvec"], current_dict["tvec"], current_dict["camera_id"], current_dict["name"] = int(image_array_parts[0]), [float(k) for k in image_array_parts[1:5]], [float(k) for k in image_array_parts[5:8]], int(image_array_parts[8]), image_array_parts[9]
                rest_image_array_parts = line2.split(" ")
                current_dict["name"] = current_dict["name"][:-1]
                current_dict["xys"], current_dict["point3D_ids"] = [[float(k) for k in rest_image_array_parts[0::3]], [float(l) for l in rest_image_array_parts[1::3]]], rest_image_array_parts[2::3]
                current_dict["point3D_ids"][-1] = current_dict["point3D_ids"][-1][:-1]
                current_dict["point3D_ids"] = [int(k) for k in current_dict["point3D_ids"]]
                image_array.append(current_dict)
            count+=2
    return image_array

### THESE WERE ADDED HERE FOR FORMATTING FROM TXT FILES GENERATED BY HLOC NOTEBOOK ###

def load_colmap_data(realdir):
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    print(camdata)
    return camdata
    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    print("their_mats", c2w_mats, c2w_mats.shape)

    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    print("their poses", poses)
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)
    counter = 0
    for pose in poses:
        np.savetxt('poses'+str(counter)+'.txt', pose)
        counter+=1
    return poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            print(ind, len(cams), poses, "what")
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)

        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


### CHANGED VERSIONS OF EXISTING METHODS TO WORK WITH TXT FILES ###

def load_colmap_data_new(realdir):

    camdata = format_cam_data(os.path.join(realdir, 'cameras.txt'))

    cam = camdata[0]
    print('Cameras', len(cam), cam)

    h, w, f = int(cam['height']), int(cam['width']), float(cam['params'][0])

    hwf = np.array([h, w, f]).reshape([3, 1])

    imdata = format_image_data(os.path.join(realdir, 'images.txt'))

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [k['name'] for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    for k in imdata:
        im = k
        R = qvec2rotmat_newformat(np.array(im["qvec"]))
        t = np.array(im['tvec']).reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)


    pts3d = format_point_data(os.path.join(realdir, 'points3D.txt'))

    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
                           1)
    return poses, pts3d, perm


def save_poses_new(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(k['xyz'])
        cams = [0] * poses.shape[-1]
        for ind in k['image_ids']:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed') ### <-- THIS MOST LIKELY MEANS THAT NOT ALL IMAGES WERE REGISTERED ###
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)

        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

### CHANGED VERSIONS OF EXISTING METHODS TO WORK WITH TXT FILES ###

def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False, 
                                                 anti_aliasing=True, anti_aliasing_sigma=None)
        
        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))
            



def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    
def gen_poses(basedir, match_type, factors=None):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []

    ### THIS SECTION IS NOT NEEDED, COLMAP RUNS THROUGH THE HLOC FILES ###
    # if not all([f in files_had for f in files_needed]):
    #     print( 'Need to run COLMAP' )
    #     run_colmap(basedir, match_type)
    # else:
    #     print('Don\'t need to run COLMAP')
    ### THIS SECTION IS NOT NEEDED, COLMAP RUNS THROUGH THE HLOC FILES ###
    print( 'Post-colmap')

    ### SWAP BETWEEN OLD LOADING AND NEW LOADING ###
    # poses, pts3d, perm = load_colmap_data(basedir)
    # save_poses(basedir, poses, pts3d, perm)
    poses, pts3d, perm = load_colmap_data_new(basedir)
    save_poses_new(basedir, poses, pts3d, perm)
    ### SWAP BETWEEN OLD LOADING AND NEW LOADING ###
    if factors is not None:
        print( 'Factors:', factors)
        minify(basedir, factors)
    
    print( 'Done with imgs2poses' )
    
    return True
    
