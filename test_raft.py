import onnxruntime as ort
import cv2
import numpy as np

MODEL = "models/optical_flow_estimation_raft_2023aug.onnx"

# img2 is img1 shifted right by 10 px => flow from img1 to img2 is approx (10, 0)
img1 = np.random.randint(0, 256, (360, 480, 3), dtype=np.uint8)
img2 = np.roll(img1, 10, axis=1)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
print("=== RAFT full model I/O ===")
print("Inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])

def preprocess(img):
    img = cv2.resize(img, (480, 360))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, axis=0).astype(np.float32)

inp1 = preprocess(img1)
inp2 = preprocess(img2)
outs = sess.run(None, {"0": inp1, "1": inp2})
print("num outputs:", len(outs))
out_names = [o.name for o in sess.get_outputs()]
for name, arr in zip(out_names, outs):
    print("OUT", name, arr.shape, arr.dtype)
flow = outs[-1]  # [1,2,360,480] upsampled flow
flow = np.squeeze(flow).transpose(1, 2, 0)  # [360,480,2]
print("flow shape:", flow.shape, "mean u,v:", flow[...,0].mean(), flow[...,1].mean())

def warp(img, disp):
    # disp (u,v): how far content is displaced; output(q) = img(q - disp)
    h, w = img.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (grid_x - disp[..., 0]).astype(np.float32)
    map_y = (grid_y - disp[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Build intermediate frame at time t
t = 0.5
warp_a = warp(img1, flow * t)        # A content displaced by t*flow
warp_b = warp(img2, flow * (-(1-t))) # B content displaced opposite
alpha = 1 - t
out = cv2.addWeighted(warp_a, alpha, warp_b, 1-alpha, 0)
print("warp_a shape:", warp_a.shape)
print("out shape:", out.shape)

# Sanity: at t=0.5 with shift-right-by-10, both warps should show img1 shifted right by 5
shifted = np.roll(img1, 5, axis=1)
diff_a = np.abs(warp_a.astype(np.int32) - shifted.astype(np.int32)).mean()
diff_b = np.abs(warp_b.astype(np.int32) - shifted.astype(np.int32)).mean()
print("warp_a vs img1+5 shift mean abs diff:", diff_a)
print("warp_b vs img1+5 shift mean abs diff:", diff_b)
print("RAFT inference + warp math OK")
