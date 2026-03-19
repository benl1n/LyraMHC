import torch.nn as nn


def weight_initial(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
