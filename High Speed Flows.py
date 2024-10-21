import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 120)  # 2 inputs: x, t
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 120)
        self.fc4 = nn.Linear(120, 120)
        self.fc5 = nn.Linear(120, 3)  # 3 outputs: density, velocity, pressure

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return self.fc5(x)  # Output: [ρ, u, p]

def data_loss(predicted, actual):
    return torch.mean((predicted - actual) ** 2)


def physics_loss(nn_output, x, t):
    # Assuming nn_output = [ρ, u, p] and applying Euler equations
    rho, u, p = nn_output[:, 0], nn_output[:, 1], nn_output[:, 2]
    rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Mass, momentum, energy conservation constraints
    mass_conservation = rho_t + (rho * u).sum()  # Simplified for illustration
    momentum_conservation = u_t + (rho * u ** 2 + p).sum()
    energy_conservation = p_t + (u * (rho * u ** 2 + p)).sum()

    return torch.mean(mass_conservation ** 2 + momentum_conservation ** 2 + energy_conservation ** 2)


def total_loss(predicted, actual, nn_output, x, t, lambda_physics=0.1):
    data_loss_value = data_loss(predicted, actual)
    physics_loss_value = physics_loss(nn_output, x, t)
    return data_loss_value + lambda_physics * physics_loss_value
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Forward pass
    nn_output = model(input_data)  # input_data = [x, t]
    predicted = nn_output[:, 0]  # Assuming first output is density, modify as needed

    # Compute loss
    loss = total_loss(predicted, actual_data, nn_output, x, t)
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
gamma = nn.Parameter(torch.tensor([1.4]))  # Learnable parameter for adiabatic index

