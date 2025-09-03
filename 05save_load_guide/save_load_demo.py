import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import onnx
import onnxruntime as ort

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


# def save():
#     model = Net()
#     # input = torch.randn(1, 1, 28, 28)
#     # output = model(input)
#     torch.save(model, "minist.pt")


def load():
    model = torch.load("minist.pt", weights_only=False)
    input = torch.rand(1, 1, 28, 28)
    output = model(input)
    print(output.shape)

# def save_demo_v1():
#     model = Net()
#     input = torch.rand(1, 1, 28, 28)
#     output = model(input)
#     torch.save(model, "mnist.pt")  # 4.6M : 保存
#
#
# def load_demo_v1():
#     model = torch.load("mnist.pt")
#     input = torch.rand(1, 1, 28, 28)
#     output = model(input)
#     print(f"output shape: {output.shape}")

# def save_ckpt_demo():
#     model = Net()
#     optimizer = optim.Adam(model.parameters(), lr=0.02)
#     loss = torch.tensor([0.14])
#     epoch = 10
#     checkpoint = {#"epoch": epoch,
#                   # "loss": loss.item(),
#                   # "optim_state_dict": optimizer.state_dict(),
#                   "model_state_dict": model.state_dict()}
#     torch.save(checkpoint, "mnist.ckpt")

# def load_ckpt_demo():
#     checkpoint = torch.load("mnist.ckpt")
#     model = Net()
#     optimizer = optim.Adam(model.parameters(), lr=0.02)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optim_state_dict"])
#     loss = checkpoint["loss"]
#     epoch = checkpoint["epoch"]
#     input = torch.rand(1, 1, 28, 28)
#     output = model(input)
#     print(f"output shape: {output.shape}")


# def save_trace_model():
#     model = Net()
#     # model.eval()
#     trace_model = torch.jit.trace(model, torch.randn(1, 1, 28, 28))
#     trace_model.save("trace_model.pt")


def onnx_demo():
    model = Net()
    # torch.onnx.export(model, torch.randn(1, 1, 28, 28), "mniist_onnx.onnx")
    onnx_model = torch.onnx.export(model, torch.randn(1, 1, 28, 28), dynamo=True)
    onnx_model.save("miniist_onnx.onnx")


def onnx_ifer():
    input = torch.randn(1, 1, 28, 28)
    ort_session = ort.InferenceSession("mnist_onnx.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print(ort_outputs[0])

def load_trace_moidel():
    mm = torch.jit.load("trace_model.pt")
    output = mm(torch.randn(1, 1, 28, 28))

    print(output.shape)


if __name__ == "__main__":
    # save()
    # load()
    # save_ckpt_demo()
    # load_ckpt_demo()
    # save_trace_model()
    # load_trace_moidel()
    # onnx_demo()
    onnx_ifer()
    print("run save_load_demo.py successful")
