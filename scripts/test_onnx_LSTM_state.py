import numpy as np
import onnxruntime as ort

path = "logs/adam_pro_12dof/exported/onnx/policy_lstm.onnx"
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

obs = np.zeros((1, 47), dtype=np.float32)
h = np.zeros((1, 1, 128), dtype=np.float32)
c = np.zeros((1, 1, 128), dtype=np.float32)

for i in range(5):
    action, h, c = sess.run(
        ["action", "h_out", "c_out"],
        {
            "obs": obs,
            "h_in": h,
            "c_in": c,
        },
    )
    print(i, "action mean:", action.mean(), "h norm:", np.linalg.norm(h), "c norm:", np.linalg.norm(c))
    