import face_alignment
import numpy as np
import scipy.io as io
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import sys

model_shape = io.loadmat('data/Model_Shape.mat')
model_exp = io.loadmat('data/Model_Exp.mat')
data = io.loadmat('data/sigma_exp.mat')

def preds_to_shape(preds):
    # paras = torch.mul(preds[:228, :], label_std[:199+29, :])
    alpha = np.reshape(preds[:199], [199,1]) * np.reshape(model_shape['sigma'], [199,1])
    beta = np.reshape(preds[199:228], [29, 1]) * 1.0/(1000.0 * np.reshape(data['sigma_exp'], [29, 1]))
    face_shape = np.matmul(model_shape['w'], alpha) + np.matmul(model_exp['w_exp'], beta) + model_shape['mu_shape']
    face_shape = face_shape.reshape(-1, 3)
    
    #obj['v'] = face_shape
    #obj['tri'] = model_shape['tri'].transpose()
    return [face_shape, model_shape['tri'].astype(np.int64).transpose() - 1]
    
    
def crop_image(image, res=224):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
    pts = fa.get_landmarks(np.array(image))
    if len(pts) < 1:
        assert "No face detected!"
    pts = np.array(pts[0]).astype(np.int32)
        
    h = image.size[1]
    w = image.size[0]
        # x-width-pts[0,:], y-height-pts[1,:]
    x_max = np.max(pts[:68, 0])
    x_min = np.min(pts[:68, 0])
    y_max = np.max(pts[:68, 1])
    y_min = np.min(pts[:68, 1])
    bbox = [y_min, x_min, y_max, x_max]
    # c (cy, cx)
    c = [bbox[2] - (bbox[2] - bbox[0]) / 2, bbox[3] - (bbox[3] - bbox[1]) / 2.0]
    c[0] = c[0] - (bbox[2] - bbox[0]) * 0.12
    s = (max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5).astype(np.int32)
    old_bb = np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2]).astype(np.int32)
    crop_img = Image.new('RGB', (s, s))
    #crop_img = torch.zeros(image.shape[0], s, s, dtype=torch.float32)

    shift_x = 0 - old_bb[1]
    shift_y = 0 - old_bb[0]
    old_bb = np.array([max(0, old_bb[0]), max(0, old_bb[1]),
              min(h, old_bb[2]), min(w, old_bb[3])]).astype(np.int32)
    hb = old_bb[2] - old_bb[0]
    wb = old_bb[3] - old_bb[1]
    new_bb = np.array([max(0, shift_y), max(0, shift_x), max(0, shift_y) + hb, max(0, shift_x) + wb]).astype(np.int32)
    cache = image.crop((old_bb[1], old_bb[0], old_bb[3], old_bb[2]))
    crop_img.paste(cache, (new_bb[1], new_bb[0], new_bb[3], new_bb[2]))
    crop_img = crop_img.resize((res, res), Image.BICUBIC)
    return crop_img

def write_ply(filename, points=None, mesh=None, colors=None, as_text=True):
    points = pd.DataFrame(points, columns=["x", "y", "z"])
    mesh = pd.DataFrame(mesh, columns=["v1", "v2", "v3"])
    if colors is not None:
        colors = pd.DataFrame(colors, columns=["red", "green", "blue"])
        points = pd.concat([points, colors], axis=1)
    """
 
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
 
    Returns
    -------
    boolean
        True if no problems
 
    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        # open in binary/append to use tofile
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True

def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + str(df.columns.values[i]))

    return element
