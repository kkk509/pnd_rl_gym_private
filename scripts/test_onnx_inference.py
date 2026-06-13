import numpy as np
import onnxruntime as ort

path = "logs/adam_pro_12dof/exported/onnx/policy_lstm.onnx"

sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

obs = np.zeros((1, 47), dtype=np.float32)
h = np.zeros((1, 1, 128), dtype=np.float32)
c = np.zeros((1, 1, 128), dtype=np.float32)

action, h_out, c_out = sess.run(
    ["action", "h_out", "c_out"],
    {
        "obs": obs,
        "h_in": h,
        "c_in": c,
    },
)

print("action shape:", action.shape)
print("h_out shape:", h_out.shape)
print("c_out shape:", c_out.shape)
print("action:", action)
print("has nan:", np.isnan(action).any())