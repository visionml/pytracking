import torch
import torch.nn as nn
from ltr.models.layers.blocks import conv_block


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros'):
        " Referenced from https://github.com/happyjin/ConvGRU-pytorch"
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        if padding_mode == 'zeros':
            if not isinstance(kernel_size, (list, tuple)):
                kernel_size = (kernel_size, kernel_size)

            padding = kernel_size[0] // 2, kernel_size[1] // 2
            self.conv_reset = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
            self.conv_update = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)

            self.conv_state_new = nn.Conv2d(input_dim+hidden_dim, self.hidden_dim, kernel_size, padding=padding)
        else:
            self.conv_reset = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                         padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                         padding_mode=padding_mode)

            self.conv_update = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                          padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                          padding_mode=padding_mode)

            self.conv_state_new = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1,
                                             padding=int(kernel_size // 2), batch_norm=False, relu=False,
                                             padding_mode=padding_mode)

    def forward(self, input, state_cur):
        input_state_cur = torch.cat([input, state_cur], dim=1)

        reset_gate = torch.sigmoid(self.conv_reset(input_state_cur))
        update_gate = torch.sigmoid(self.conv_update(input_state_cur))

        input_state_cur_reset = torch.cat([input, reset_gate*state_cur], dim=1)
        state_new = torch.tanh(self.conv_state_new(input_state_cur_reset))

        state_next = (1.0 - update_gate) * state_cur + update_gate * state_new
        return state_next
