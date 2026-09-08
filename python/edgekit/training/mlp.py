from edgekit.models.mlp import MLP, sigmoid_derivative


def train_mlp(
    model: MLP,
    dataset,
    epochs: int = 1000,
    learning_rate: float = 0.1,
    momentum: float = 0.9,
) -> None:
    velocity_input_hidden = [[0.0] * model.n_inputs for _ in range(model.n_hidden)]
    velocity_bias_hidden = [0.0] * model.n_hidden
    velocity_hidden_output = [[0.0] * model.n_hidden for _ in range(model.n_outputs)]
    velocity_bias_output = [0.0] * model.n_outputs

    for _ in range(epochs):
        for x, target in dataset:
            hidden, output = model.forward(x)

            output_deltas = [
                (target - output[o]) * sigmoid_derivative(output[o])
                for o in range(model.n_outputs)
            ]

            hidden_deltas = [
                sum(
                    output_deltas[o] * model.weights_hidden_output[o][h]
                    for o in range(model.n_outputs)
                )
                * sigmoid_derivative(hidden[h])
                for h in range(model.n_hidden)
            ]

            for o in range(model.n_outputs):
                for h in range(model.n_hidden):
                    velocity_hidden_output[o][h] = (
                        momentum * velocity_hidden_output[o][h]
                        + learning_rate * output_deltas[o] * hidden[h]
                    )
                    model.weights_hidden_output[o][h] += velocity_hidden_output[o][h]

                velocity_bias_output[o] = (
                    momentum * velocity_bias_output[o] + learning_rate * output_deltas[o]
                )
                model.bias_output[o] += velocity_bias_output[o]

            for h in range(model.n_hidden):
                for i in range(model.n_inputs):
                    velocity_input_hidden[h][i] = (
                        momentum * velocity_input_hidden[h][i]
                        + learning_rate * hidden_deltas[h] * x[i]
                    )
                    model.weights_input_hidden[h][i] += velocity_input_hidden[h][i]

                velocity_bias_hidden[h] = (
                    momentum * velocity_bias_hidden[h] + learning_rate * hidden_deltas[h]
                )
                model.bias_hidden[h] += velocity_bias_hidden[h]