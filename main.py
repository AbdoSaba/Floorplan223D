import cv2
import numpy as np
import onnxruntime as ort
import sys
import os
import pyvista as pv
import trimesh
import warnings

warnings.filterwarnings('ignore')

# Get the directory of the current script to construct relative path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "best.onnx")

CLASSES = ["none", "door", "wall", "window"]
CONF_THRESH = 0.3
INPUT_SHAPE = (616, 616)

def get_img(path):
    if not os.path.exists(path): sys.exit(f"File not found: {path}")
    img = cv2.imread(path)
    return img if img is not None else sys.exit("Invalid image")

def check_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}.\nPlease ensure weights/best.onnx exists.")


def prepare_blob(img):
    h, w = img.shape[:2]
    scale = min(INPUT_SHAPE[0]/h, INPUT_SHAPE[1]/w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((INPUT_SHAPE[0], INPUT_SHAPE[1], 3), 0, dtype=np.uint8)
    dw, dh = (INPUT_SHAPE[1] - nw) // 2, (INPUT_SHAPE[0] - nh) // 2
    padded[dh:dh+nh, dw:dw+nw] = resized
    norm = (padded.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    blob = np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0).astype(np.float32)
    return blob, scale, (dw, dh), (h, w)

def detect(session, blob, scale, pad, orig_dim):
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    boxes, scores = outputs[0][0], 1 / (1 + np.exp(-outputs[1][0]))
    flat_scores = scores.reshape(-1)
    indices = np.argsort(-flat_scores)
    indices = indices[flat_scores[indices] > CONF_THRESH]
    res = []
    for idx in indices:
        cls = idx % 4
        if cls == 0: continue
        box_idx = idx // 4
        cx, cy, w, h = boxes[box_idx]
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        x1 = (x1 * INPUT_SHAPE[1] - pad[0]) / scale
        y1 = (y1 * INPUT_SHAPE[0] - pad[1]) / scale
        x2 = (x2 * INPUT_SHAPE[1] - pad[0]) / scale
        y2 = (y2 * INPUT_SHAPE[0] - pad[1]) / scale
        res.append([max(0, x1), max(0, y1), min(orig_dim[1], x2), min(orig_dim[0], y2), cls])
    return res

class Builder:
    def __init__(self, real_w, real_d, img_w, img_h, wall_h, wall_t):
        self.sx = real_w / img_w
        self.sy = real_d / img_h
        self.h_px = img_h
        self.wall_h = wall_h
        self.wall_t = wall_t
        self.door_h = wall_h * 0.75
        self.win_sill = wall_h * 0.35
        self.win_top = wall_h * 0.75
        self.off_x = real_w / 2
        self.off_z = real_d / 2

    def to_3d(self, box):
        x1, y1, x2, y2 = box
        wx1, wx2 = self.off_x - (x2 * self.sx), self.off_x - (x1 * self.sx)
        wz1, wz2 = ((self.h_px - y2) * self.sy) - self.off_z, ((self.h_px - y1) * self.sy) - self.off_z
        return wx1, wz1, wx2, wz2

    def create_scene(self, dets):
        walls, holes = [], []
        for d in dets:
            coords = self.to_3d(d[:4])
            if CLASSES[d[4]] == 'wall':
                wx1, wz1, wx2, wz2 = coords
                if (wx2 - wx1) > (wz2 - wz1):
                    mid_z = (wz1 + wz2) / 2
                    wz1, wz2 = mid_z - self.wall_t/2, mid_z + self.wall_t/2
                else:
                    mid_x = (wx1 + wx2) / 2
                    wx1, wx2 = mid_x - self.wall_t/2, mid_x + self.wall_t/2
                walls.append({'box': (wx1, wz1, wx2, wz2), 'cuts': []})
            else:
                y_min = 0 if CLASSES[d[4]] == 'door' else self.win_sill
                y_max = self.door_h if CLASSES[d[4]] == 'door' else self.win_top
                holes.append({'box': coords, 'y': (y_min, y_max)})

        for h in holes:
            hx1, hz1, hx2, hz2 = h['box']
            for w in walls:
                wx1, wz1, wx2, wz2 = w['box']
                if hx1 < wx2 and hx2 > wx1 and hz1 < wz2 and hz2 > wz1:
                    w['cuts'].append({'x1': max(wx1, hx1), 'z1': max(wz1, hz1),
                                    'x2': min(wx2, hx2), 'z2': min(wz2, hz2),
                                    'y': h['y']})
        meshes = []
        for w in walls:
            wx1, wz1, wx2, wz2 = w['box']
            is_horz = (wx2 - wx1) > (wz2 - wz1)
            
            if not w['cuts']:
                meshes.append(self.box(wx1, wz1, 0, wx2, wz2, self.wall_h))
                continue
            
            w['cuts'].sort(key=lambda c: c['x1'] if is_horz else c['z1'])
            curr = wx1 if is_horz else wz1
            
            for c in w['cuts']:
                if is_horz:
                    if curr < c['x1']: 
                        meshes.append(self.box(curr, wz1, 0, c['x1'], wz2, self.wall_h))
                    if c['y'][1] < self.wall_h: 
                        meshes.append(self.box(c['x1'], wz1, c['y'][1], c['x2'], wz2, self.wall_h))
                    if c['y'][0] > 0: 
                        meshes.append(self.box(c['x1'], wz1, 0, c['x2'], wz2, c['y'][0]))
                    curr = max(curr, c['x2'])
                else:
                    if curr < c['z1']: 
                        meshes.append(self.box(wx1, curr, 0, wx2, c['z1'], self.wall_h))
                    if c['y'][1] < self.wall_h: 
                        meshes.append(self.box(wx1, c['z1'], c['y'][1], wx2, c['z2'], self.wall_h))
                    if c['y'][0] > 0: 
                        meshes.append(self.box(wx1, c['z1'], 0, wx2, c['z2'], c['y'][0]))
                    curr = max(curr, c['z2'])
            
            if is_horz and curr < wx2: 
                meshes.append(self.box(curr, wz1, 0, wx2, wz2, self.wall_h))
            if not is_horz and curr < wz2: 
                meshes.append(self.box(wx1, curr, 0, wx2, wz2, self.wall_h))
                
        return [m for m in meshes if m]

    def box(self, x1, z1, y1, x2, z2, y2):
        w = abs(x2 - x1)
        d = abs(z2 - z1)
        h = abs(y2 - y1)
        
        if w < 0.1 or d < 0.1 or h < 0.1: return None
        
        b = trimesh.creation.box(extents=(w, h, d))
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cz = (z1 + z2) / 2
        
        b.apply_translation((cx, cy, cz))
        return b

if __name__ == "__main__":
    check_model()
    path = input("Image Path: ").strip().strip('"')
    img = get_img(path)
    oh, ow = img.shape[:2]
    
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    blob, scale, pad, _ = prepare_blob(img)
    dets = detect(sess, blob, scale, pad, (oh, ow))
    
    rw = float(input("Real Width : ") or 3000)
    rd = rw * (oh / ow)
    wh = float(input("Wall Height : ") or 450)
    wt = float(input("Wall Thickness : ") or 50)
    
    builder = Builder(rw, rd, ow, oh, wh, wt)
    meshes = builder.create_scene(dets)
    
    if meshes:
        scene = trimesh.Scene(meshes)
        
        floor = trimesh.creation.box(extents=(rw, 10, rd))
        floor.apply_translation((0, -5, 0))
        
        scene.add_geometry(floor)
        scene.export("output.obj")
        
        pl = pv.Plotter()
        pl.set_background('white')
        
        for m in meshes:
            pl.add_mesh(pv.wrap(m), color='#F5F5DC', show_edges=True, edge_color='#8B4513')
        pl.add_mesh(pv.wrap(floor), color='#404040', opacity=1.0)
        
        pl.camera_position = 'iso'
        pl.show()